using System.Data;
using System.Data.SqlClient;
using System.Linq;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

// Initialize Cn
var ctx = new MLContext();

var inferResults = 
    ctx.Auto().InferColumns(@"C:\Datasets\RestaurantScores.tsv", labelColumnName: "RiskCategory", separatorChar:'\t');

var textLoaderCols = inferResults.TextLoaderOptions.Columns;

var dbCols =
    textLoaderCols
        .Select(col =>
        {
            var dbType = MapDbType(col.DataKind);
            var source = MapSource(col.Source);
            return new DatabaseLoader.Column(col.Name, dbType, source, col.KeyCount);
        })
        .ToArray();

var sqloader = ctx.Data.CreateDatabaseLoader(dbCols);

string connectionString = @"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=C:\Datasets\RestaurantScores.mdf;Integrated Security=True;Connect Timeout=30";

string sqlCommand = "SELECT * FROM Violations";

var dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, sqlCommand);

var data = sqloader.Load(dbSource);

var dataSplit = ctx.Data.TrainTestSplit(data, testFraction: 0.2);

var pipeline =
    ctx.Auto().Featurizer(data, inferResults.ColumnInformation, outputColumnName: "Features")
        .Append(ctx.Transforms.Conversion.MapValueToKey("Label", "RiskCategory"))
        .Append(ctx.Auto().MultiClassification(useLgbm:false))
        .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

var experiment = ctx.Auto().CreateExperiment();

experiment
    .SetDataset(dataSplit)
    .SetPipeline(pipeline)
    .SetTrainingTimeInSeconds(90)
    .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MicroAccuracy, labelColumn: "Label");

var expResult = await experiment.RunAsync();

Console.WriteLine($"Micro-Accuracy: {expResult.Metric}");

DbType MapDbType(DataKind t)
{
    // This only works with strings, but could be expanded to map other data types
    return t switch
    {
        DataKind.String => DbType.String,
    };
}

DatabaseLoader.Range[] MapSource(TextLoader.Range[] textRange)
{
    var dbRanges =
        textRange
            .Select(r =>
            {
                return (r.Min == r.Max) switch
                {
                    true => new DatabaseLoader.Range(r.Min),
                    false => new DatabaseLoader.Range(r.Min,r.Max.Value)
                };
            })
            .ToArray();
    return dbRanges;
}