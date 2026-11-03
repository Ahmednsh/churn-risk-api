using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ChurnRisk.Api;

/// <summary>Runs the model trained in Python. One session is shared, because Run is safe to call concurrently.</summary>
public sealed class ChurnModel : IDisposable
{
    private readonly InferenceSession _session;

    public ChurnModel(string path)
    {
        _session = new InferenceSession(path);
    }

    /// <summary>Returns the probability that this customer churns.</summary>
    public double Predict(CustomerFeatures customer)
    {
        // The model takes one input per column, each shaped [rows, 1]. Here there is one row.
        int[] shape = [1, 1];
        var inputs = new List<NamedOnnxValue>();
        foreach (var (name, value) in customer.Numeric)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(new[] { value }, shape)));
        }
        foreach (var (name, value) in customer.Categorical)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<string>(new[] { value }, shape)));
        }

        using var results = _session.Run(inputs);
        var probabilities = results.First(result => result.Name == "probabilities").AsTensor<float>();

        // Column 0 is "stays" and column 1 is "churns".
        return probabilities[0, 1];
    }

    public void Dispose() => _session.Dispose();
}
