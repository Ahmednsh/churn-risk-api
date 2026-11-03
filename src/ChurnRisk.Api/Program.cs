using System.Diagnostics;
using System.Text.Json;
using ChurnRisk.Api;

var builder = WebApplication.CreateBuilder(args);

var modelDirectory = Path.Combine(AppContext.BaseDirectory, "model");
builder.Services.AddSingleton(_ => ModelSchema.Load(Path.Combine(modelDirectory, "schema.json")));
builder.Services.AddSingleton(_ => new ChurnModel(Path.Combine(modelDirectory, "churn_model.onnx")));
builder.Services.AddSingleton(services =>
{
    var path = services.GetRequiredService<IConfiguration>()["ChurnRisk:DbPath"]
        ?? Path.Combine(AppContext.BaseDirectory, "predictions.db");
    return new PredictionLog(path);
});

var app = builder.Build();

app.MapGet("/health", (ModelSchema schema) => Results.Ok(new { status = "ok", modelVersion = schema.ModelVersion }));

app.MapPost("/predict", (Dictionary<string, JsonElement> customer, ModelSchema schema, ChurnModel model, PredictionLog log) =>
{
    var errors = CustomerValidator.Validate(customer, schema, out var features);
    if (errors.Count > 0)
    {
        return Results.ValidationProblem(errors);
    }

    var timer = Stopwatch.StartNew();
    var probability = model.Predict(features);
    timer.Stop();

    var riskBand = RiskBands.For(probability, schema.RiskBands);
    log.Record(probability, riskBand, timer.Elapsed.TotalMilliseconds, schema.ModelVersion);

    return Results.Ok(new { probability, riskBand, modelVersion = schema.ModelVersion });
});

app.MapGet("/monitoring/summary", (ModelSchema schema, PredictionLog log) =>
{
    var summary = log.Summarize(schema.ModelVersion);
    return Results.Ok(new
    {
        modelVersion = schema.ModelVersion,
        total = summary.Total,
        meanProbability = summary.MeanProbability,
        baselineMeanProbability = schema.Monitoring.BaselineMeanProbability,
        driftAlert = MonitoringRules.IsDrifted(summary.Total, summary.MeanProbability, schema.Monitoring),
        averageLatencyMs = summary.AverageLatencyMs,
        bands = new { low = summary.Low, medium = summary.Medium, high = summary.High },
    });
});

app.Run();
