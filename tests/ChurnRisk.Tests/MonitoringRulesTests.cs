using ChurnRisk.Api;

namespace ChurnRisk.Tests;

public class MonitoringRulesTests
{
    private static readonly RiskBandLimits Limits = new(Medium: 0.3, High: 0.6);
    private static readonly MonitoringSettings Settings = new(BaselineMeanProbability: 0.27, DriftTolerance: 0.10, MinimumPredictions: 30);

    [Theory]
    [InlineData(0.0, "low")]
    [InlineData(0.2999, "low")]
    [InlineData(0.3, "medium")]
    [InlineData(0.5999, "medium")]
    [InlineData(0.6, "high")]
    [InlineData(1.0, "high")]
    public void Risk_band_follows_the_limits(double probability, string expected)
    {
        Assert.Equal(expected, RiskBands.For(probability, Limits));
    }

    [Fact]
    public void Drift_stays_quiet_until_there_are_enough_predictions()
    {
        Assert.False(MonitoringRules.IsDrifted(total: 29, meanProbability: 0.9, Settings));
    }

    [Fact]
    public void Drift_alerts_when_scores_rise_past_the_tolerance()
    {
        Assert.True(MonitoringRules.IsDrifted(total: 30, meanProbability: 0.38, Settings));
    }

    [Fact]
    public void Drift_alerts_when_scores_fall_past_the_tolerance()
    {
        Assert.True(MonitoringRules.IsDrifted(total: 30, meanProbability: 0.16, Settings));
    }

    [Fact]
    public void Drift_stays_quiet_inside_the_tolerance()
    {
        Assert.False(MonitoringRules.IsDrifted(total: 30, meanProbability: 0.36, Settings));
    }
}
