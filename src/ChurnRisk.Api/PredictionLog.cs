using System.Globalization;
using Microsoft.Data.Sqlite;

namespace ChurnRisk.Api;

public sealed record PredictionSummary(int Total, double MeanProbability, double AverageLatencyMs, int Low, int Medium, int High);

/// <summary>Records each prediction in SQLite for monitoring. It stores scores only, never customer details.</summary>
public sealed class PredictionLog
{
    private readonly string _connectionString;

    public PredictionLog(string databasePath)
    {
        _connectionString = $"Data Source={databasePath}";

        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = """
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY,
                created_utc   TEXT NOT NULL,
                probability   REAL NOT NULL,
                risk_band     TEXT NOT NULL,
                latency_ms    REAL NOT NULL,
                model_version TEXT NOT NULL
            );
            """;
        command.ExecuteNonQuery();
    }

    public void Record(double probability, string riskBand, double latencyMs, string modelVersion)
    {
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO predictions (created_utc, probability, risk_band, latency_ms, model_version)
            VALUES ($createdUtc, $probability, $riskBand, $latencyMs, $modelVersion);
            """;
        command.Parameters.AddWithValue("$createdUtc", DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture));
        command.Parameters.AddWithValue("$probability", probability);
        command.Parameters.AddWithValue("$riskBand", riskBand);
        command.Parameters.AddWithValue("$latencyMs", latencyMs);
        command.Parameters.AddWithValue("$modelVersion", modelVersion);
        command.ExecuteNonQuery();
    }

    /// <summary>Summarises one model version, because a new model has a different baseline to compare with.</summary>
    public PredictionSummary Summarize(string modelVersion)
    {
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT COUNT(*),
                   COALESCE(AVG(probability), 0),
                   COALESCE(AVG(latency_ms), 0),
                   COALESCE(SUM(risk_band = 'low'), 0),
                   COALESCE(SUM(risk_band = 'medium'), 0),
                   COALESCE(SUM(risk_band = 'high'), 0)
            FROM predictions
            WHERE model_version = $modelVersion;
            """;
        command.Parameters.AddWithValue("$modelVersion", modelVersion);

        using var reader = command.ExecuteReader();
        reader.Read();
        return new PredictionSummary(
            reader.GetInt32(0), reader.GetDouble(1), reader.GetDouble(2),
            reader.GetInt32(3), reader.GetInt32(4), reader.GetInt32(5));
    }

    private SqliteConnection Open()
    {
        var connection = new SqliteConnection(_connectionString);
        connection.Open();
        return connection;
    }
}
