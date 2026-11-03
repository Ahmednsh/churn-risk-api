using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Data.Sqlite;

namespace ChurnRisk.Tests;

/// <summary>Runs the real API in memory, with the real ONNX model and a real SQLite file.</summary>
public sealed class ApiTests : IDisposable
{
    private readonly string _dbPath = Path.Combine(Path.GetTempPath(), $"churn-test-{Guid.NewGuid():N}.db");
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public ApiTests()
    {
        _factory = CreateFactory();
        _client = _factory.CreateClient();
    }

    private WebApplicationFactory<Program> CreateFactory() =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(host => host.UseSetting("ChurnRisk:DbPath", _dbPath));

    private static List<Dictionary<string, JsonElement>> ParityCases()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "model", "parity_cases.json");
        return JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(File.ReadAllText(path))!;
    }

    private static Dictionary<string, JsonElement> ValidCustomer()
    {
        var customer = ParityCases()[0];
        customer.Remove("expectedProbability");
        return customer;
    }

    private static JsonElement Json(object value) => JsonSerializer.SerializeToElement(value);

    private async Task<JsonElement> Summary()
    {
        var response = await _client.GetAsync("/monitoring/summary");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return await response.Content.ReadFromJsonAsync<JsonElement>();
    }

    private async Task AssertRejected(Dictionary<string, JsonElement> customer, string field)
    {
        var response = await _client.PostAsJsonAsync("/predict", customer);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.True(body.GetProperty("errors").TryGetProperty(field, out _), $"No error named {field}: {body}");
    }

    [Fact]
    public async Task Health_reports_the_loaded_model_version()
    {
        var body = await _client.GetFromJsonAsync<JsonElement>("/health");

        Assert.Equal("ok", body.GetProperty("status").GetString());
        Assert.False(string.IsNullOrEmpty(body.GetProperty("modelVersion").GetString()));
    }

    [Fact]
    public async Task Predict_gives_the_same_probability_as_the_trained_model()
    {
        foreach (var testCase in ParityCases())
        {
            var expected = testCase["expectedProbability"].GetDouble();
            testCase.Remove("expectedProbability");

            var response = await _client.PostAsJsonAsync("/predict", testCase);

            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            var body = await response.Content.ReadFromJsonAsync<JsonElement>();
            Assert.Equal(expected, body.GetProperty("probability").GetDouble(), tolerance: 0.001);
        }
    }

    [Fact]
    public async Task Predict_returns_a_risk_band_and_the_model_version()
    {
        var response = await _client.PostAsJsonAsync("/predict", ValidCustomer());

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Contains(body.GetProperty("riskBand").GetString(), new[] { "low", "medium", "high" });
        Assert.False(string.IsNullOrEmpty(body.GetProperty("modelVersion").GetString()));
    }

    [Fact]
    public async Task Predict_rejects_a_category_the_model_has_never_seen()
    {
        var customer = ValidCustomer();
        customer["Contract"] = Json("Weekly");
        await AssertRejected(customer, "Contract");
    }

    [Fact]
    public async Task Predict_rejects_a_number_outside_its_limits()
    {
        var customer = ValidCustomer();
        customer["tenure"] = Json(-1);
        await AssertRejected(customer, "tenure");
    }

    [Fact]
    public async Task Predict_rejects_text_where_a_number_is_required()
    {
        var customer = ValidCustomer();
        customer["tenure"] = Json("twelve");
        await AssertRejected(customer, "tenure");
    }

    [Fact]
    public async Task Predict_rejects_a_number_where_text_is_required()
    {
        var customer = ValidCustomer();
        customer["Contract"] = Json(1);
        await AssertRejected(customer, "Contract");
    }

    [Fact]
    public async Task Predict_rejects_a_missing_field()
    {
        var customer = ValidCustomer();
        customer.Remove("PaymentMethod");
        await AssertRejected(customer, "PaymentMethod");
    }

    [Fact]
    public async Task Predict_rejects_a_field_it_does_not_use()
    {
        var customer = ValidCustomer();
        customer["customerID"] = Json("7590-VHVEG");
        await AssertRejected(customer, "customerID");
    }

    [Fact]
    public async Task Predict_rejects_malformed_json()
    {
        var content = new StringContent("{not json", System.Text.Encoding.UTF8, "application/json");

        var response = await _client.PostAsync("/predict", content);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Summary_is_empty_before_any_prediction()
    {
        var summary = await Summary();

        Assert.Equal(0, summary.GetProperty("total").GetInt32());
        Assert.False(summary.GetProperty("driftAlert").GetBoolean());
    }

    [Fact]
    public async Task Summary_counts_predictions_by_band_and_averages_their_probability()
    {
        var probabilities = new List<double>();
        foreach (var testCase in ParityCases().Take(5))
        {
            testCase.Remove("expectedProbability");
            var response = await _client.PostAsJsonAsync("/predict", testCase);
            var body = await response.Content.ReadFromJsonAsync<JsonElement>();
            probabilities.Add(body.GetProperty("probability").GetDouble());
        }

        var summary = await Summary();

        Assert.Equal(5, summary.GetProperty("total").GetInt32());
        Assert.Equal(probabilities.Average(), summary.GetProperty("meanProbability").GetDouble(), tolerance: 0.000001);
        var bands = summary.GetProperty("bands");
        Assert.Equal(5, bands.GetProperty("low").GetInt32() + bands.GetProperty("medium").GetInt32() + bands.GetProperty("high").GetInt32());
        Assert.True(summary.GetProperty("averageLatencyMs").GetDouble() > 0);
    }

    [Fact]
    public async Task Rejected_requests_are_not_counted()
    {
        var customer = ValidCustomer();
        customer["tenure"] = Json(-1);
        await _client.PostAsJsonAsync("/predict", customer);

        var summary = await Summary();

        Assert.Equal(0, summary.GetProperty("total").GetInt32());
    }

    [Fact]
    public async Task Predictions_survive_a_restart()
    {
        await _client.PostAsJsonAsync("/predict", ValidCustomer());

        using var restarted = CreateFactory();
        var body = await restarted.CreateClient().GetFromJsonAsync<JsonElement>("/monitoring/summary");

        Assert.Equal(1, body.GetProperty("total").GetInt32());
    }

    [Fact]
    public async Task The_database_stores_no_customer_details()
    {
        await _client.PostAsJsonAsync("/predict", ValidCustomer());

        using var connection = new SqliteConnection($"Data Source={_dbPath}");
        connection.Open();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT name FROM pragma_table_info('predictions') ORDER BY cid";
        var columns = new List<string>();
        using (var reader = command.ExecuteReader())
        {
            while (reader.Read())
            {
                columns.Add(reader.GetString(0));
            }
        }

        Assert.Equal(new[] { "id", "created_utc", "probability", "risk_band", "latency_ms", "model_version" }, columns);
    }

    public void Dispose()
    {
        _client.Dispose();
        _factory.Dispose();
        SqliteConnection.ClearAllPools();
        foreach (var path in new[] { _dbPath, _dbPath + "-wal", _dbPath + "-shm" })
        {
            File.Delete(path);
        }
    }
}
