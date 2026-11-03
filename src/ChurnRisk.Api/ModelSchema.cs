using System.Text.Json;

namespace ChurnRisk.Api;

public sealed record NumericLimits(double Min, double Max);

public sealed record RiskBandLimits(double Medium, double High);

public sealed record MonitoringSettings(double BaselineMeanProbability, double DriftTolerance, int MinimumPredictions);

/// <summary>What training exported about the model: its inputs, their allowed values, and its risk bands.</summary>
public sealed record ModelSchema(
    string ModelVersion,
    string ModelName,
    Dictionary<string, NumericLimits> Numeric,
    Dictionary<string, string[]> Categorical,
    RiskBandLimits RiskBands,
    MonitoringSettings Monitoring)
{
    public static ModelSchema Load(string path)
    {
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web);
        return JsonSerializer.Deserialize<ModelSchema>(File.ReadAllText(path), options)
            ?? throw new InvalidDataException($"{path} is empty");
    }
}
