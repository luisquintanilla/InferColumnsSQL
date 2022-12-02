using System.Data;
using System.Data.SqlClient;
using System.Linq;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

// Initialize MLContext
var ctx = new MLContext();

// Infer column information using tsv file
var inferResults = 
    ctx.Auto().InferColumns(@"C:\Datasets\RestaurantScores.tsv", labelColumnName: "RiskCategory", separatorChar:'\t');

// Get columns
var textLoaderCols = inferResults.TextLoaderOptions.Columns;

// Map TextLoader.Columns to DatabaseLoader.Columns
var dbCols = ConvertToDbColumn(textLoaderCols);

// Create database loader
var sqloader = ctx.Data.CreateDatabaseLoader(dbCols);

// Define connection string
string connectionString = @"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=C:\Datasets\RestaurantScores.mdf;Integrated Security=True;Connect Timeout=30";

// Define SQL Query
string sqlCommand = "SELECT * FROM Violations";

// Initialize DB Source
var dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, sqlCommand);

// Load data from mdf file
var data = sqloader.Load(dbSource);

// Split data into 80% train / 20% validation dataset
var dataSplit = ctx.Data.TrainTestSplit(data, testFraction: 0.2);

// Define pipeline using Featurizer
var pipeline =
    ctx.Auto().Featurizer(data, inferResults.ColumnInformation, outputColumnName: "Features")
        .Append(ctx.Transforms.Conversion.MapValueToKey("Label", "RiskCategory"))
        .Append(ctx.Auto().MultiClassification(useLgbm:false))
        .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

// Create AutoML experiment
var experiment = ctx.Auto().CreateExperiment();

// Configure AutoML experiment
experiment
    .SetDataset(dataSplit)
    .SetPipeline(pipeline)
    .SetTrainingTimeInSeconds(90)
    .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MicroAccuracy, labelColumn: "Label");

// Run AutoML experiment
var expResult = await experiment.RunAsync();

// Print out evaluation metric (MicroAccuracy)
Console.WriteLine($"Micro-Accuracy: {expResult.Metric}");

// Utility function to map DataKind to DbType
DbType MapDbType(DataKind t)
{
    // This only works with strings, but could be expanded to map other data types
    return t switch
    {
        DataKind.String => DbType.String,
    };
}

// Utility function to map column index information
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

// Utility function to map columns from TextLoader to DatabaseLoader
DatabaseLoader.Column[] ConvertToDbColumn(TextLoader.Column[] textLoaderColumns)
{
    return textLoaderColumns
        .Select(col =>
        {
            var dbType = MapDbType(col.DataKind);
            var source = MapSource(col.Source);
            return new DatabaseLoader.Column(col.Name, dbType, source, col.KeyCount);
        })
        .ToArray();
}