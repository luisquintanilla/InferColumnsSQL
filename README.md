# ML.NET InferColumns + AutoML with SQL DB

This sample shows how to use the AutoML InferColumns, Featurizer, and AutoML API to categorize restaurant health violations using a SQL data source.

## Data

> The data set used to train and evaluate the machine learning model is originally from the [San Francisco Department of Public Health Restaurant Safety Scores](https://www.sfdph.org/dph/EH/Food/score/default.asp). For convenience, the dataset has been condensed to only include the columns relevant to train the model and make predictions. Visit the following website to learn more about the [dataset](https://data.sfgov.org/Health-and-Social-Services/Restaurant-Scores-LIVES-Standard/pyih-qa8i?row_index=0).

[Download the Restaurant Safety Scores dataset](https://github.com/dotnet/machinelearning-samples/raw/main/samples/modelbuilder/MulticlassClassification_RestaurantViolations/RestaurantScores.zip) and unzip it.

Each row in the dataset contains information regarding violations observed during an inspection from the Health Department and a risk assessment of the threat those violations present to public health and safety.

| InspectionType | ViolationDescription | RiskCategory |
| --- | --- | --- |
| Routine - Unscheduled | Inadequately cleaned or sanitized food contact surfaces | Moderate Risk |
| New Ownership | High risk vermin infestation | High Risk |
| Routine - Unscheduled | Wiping cloths not clean or properly stored or inadequate sanitizer | Low Risk |

- **InspectionType**: the type of inspection. This can either be a first-time inspection for a new establishment, a routine inspection, a complaint inspection, and many other types.
- **ViolationDescription**: a description of the violation found during inspection.
- **RiskCategory**: the risk severity a violation poses to public health and safety.

There are two files in the directory:

- **RestaurantScore.mdf** - Database file used for training
- **RestaurantScore.tsv** - Text file containing the same information as **RestaurantScore.mdf**. Only used to infer column information. 