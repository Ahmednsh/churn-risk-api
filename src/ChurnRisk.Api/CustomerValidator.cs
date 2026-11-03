using System.Text.Json;

namespace ChurnRisk.Api;

/// <summary>A customer whose values have all passed validation, ready for the model.</summary>
public sealed record CustomerFeatures(Dictionary<string, float> Numeric, Dictionary<string, string> Categorical);

public static class CustomerValidator
{
    /// <summary>Returns one error per bad field. An empty result means <paramref name="features"/> can be used.</summary>
    public static Dictionary<string, string[]> Validate(
        Dictionary<string, JsonElement> customer, ModelSchema schema, out CustomerFeatures features)
    {
        var errors = new Dictionary<string, string[]>();
        var numeric = new Dictionary<string, float>();
        var categorical = new Dictionary<string, string>();

        foreach (var (name, limits) in schema.Numeric)
        {
            if (!customer.TryGetValue(name, out var value))
            {
                errors[name] = ["This field is required."];
            }
            else if (value.ValueKind != JsonValueKind.Number)
            {
                errors[name] = ["This field must be a number."];
            }
            else if (value.GetDouble() < limits.Min || value.GetDouble() > limits.Max)
            {
                errors[name] = [$"This field must be between {limits.Min} and {limits.Max}."];
            }
            else
            {
                numeric[name] = (float)value.GetDouble();
            }
        }

        foreach (var (name, allowed) in schema.Categorical)
        {
            if (!customer.TryGetValue(name, out var value))
            {
                errors[name] = ["This field is required."];
            }
            else if (value.ValueKind != JsonValueKind.String || !allowed.Contains(value.GetString()))
            {
                errors[name] = [$"This field must be one of: {string.Join(", ", allowed)}."];
            }
            else
            {
                categorical[name] = value.GetString()!;
            }
        }

        // Fields the model does not use are refused, so that nobody sends personal data here by accident.
        foreach (var name in customer.Keys)
        {
            if (!schema.Numeric.ContainsKey(name) && !schema.Categorical.ContainsKey(name))
            {
                errors[name] = ["This field is not used by the model."];
            }
        }

        features = new CustomerFeatures(numeric, categorical);
        return errors;
    }
}
