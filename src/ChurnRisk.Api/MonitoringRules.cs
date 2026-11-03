namespace ChurnRisk.Api;

public static class RiskBands
{
    public static string For(double probability, RiskBandLimits limits)
    {
        if (probability >= limits.High)
        {
            return "high";
        }
        return probability >= limits.Medium ? "medium" : "low";
    }
}

public static class MonitoringRules
{
    /// <summary>
    /// A simple signal that live customers no longer look like the training data: the average score
    /// has moved away from what the model gave on training data. It needs a minimum number of predictions,
    /// because a handful of customers can move an average a long way.
    /// </summary>
    public static bool IsDrifted(int total, double meanProbability, MonitoringSettings settings)
    {
        if (total < settings.MinimumPredictions)
        {
            return false;
        }
        return Math.Abs(meanProbability - settings.BaselineMeanProbability) > settings.DriftTolerance;
    }
}
