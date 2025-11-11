# Campus Demand Forecasting with Amazon SageMaker

This notebook implements a time series forecasting solution using Amazon SageMaker's DeepAR algorithm to predict occupancy rates for educational resources (libraries, computer labs, study rooms, etc.).

## Overview

The solution trains a deep learning model to forecast resource occupancy 24 hours ahead, helping educational institutions optimize resource allocation and improve student experience.

## Features

- **Automated Feature Engineering**: Creates lag features (1h, 24h, 168h) and rolling averages
- **Multi-Resource Forecasting**: Trains a single model that learns patterns across all resources
- **Probabilistic Predictions**: Provides confidence intervals (10th, 50th, 90th percentiles)
- **Visual Analytics**: Generates plots comparing actual vs predicted occupancy rates
- **Performance Metrics**: Calculates MAE, RMSE, R², and MAPE for model evaluation

## Data Requirements

The notebook expects a parquet file with the following columns:

### Required Columns
- `usage_id` - Unique identifier for each usage record
- `timestamp` - Date and time of observation
- `resource_id` - Unique identifier for the resource
- `current_occupancy` - Number of people using the resource
- `occupancy_rate` - Proportion of capacity being used (target variable)
- `day_of_week` - Day of the week
- `is_exam_period` - Boolean indicating exam period
- `is_peak_hour` - Boolean indicating peak hours

### Resource Metadata
- `resource_type` - Type of resource (e.g., LibraryRoom, ComputerLab)
- `name` - Resource name
- `location` - Resource location
- `total_capacity` - Maximum capacity
- `availability_hours` - Operating hours

### Student Context (Optional)
- `total_free_students` - Number of students without classes
- `total_busy_students` - Number of students in classes

### Lag Features (Auto-generated)
- `occupancy_rate_lag_1h` - Occupancy 1 hour ago
- `occupancy_rate_lag_24h` - Occupancy 24 hours ago
- `occupancy_rate_lag_168h` - Occupancy 1 week ago
- `occupancy_rate_avg_24h` - 24-hour rolling average

## Setup

### Prerequisites
- Amazon SageMaker notebook instance or SageMaker Studio
- IAM role with permissions for:
  - S3 read/write access
  - SageMaker training and hosting
- Python 3.8+

### Required Libraries
```python
pandas
numpy
boto3
sagemaker
matplotlib
scikit-learn
```

## Usage

### 1. Configure Data Path
Update the S3 path to your parquet file:
```python
s3_path = 's3://your-bucket/path/to/data.parquet'
```

### 2. Run the Notebook
Execute cells sequentially:
1. **Setup and Imports** - Initialize SageMaker session
2. **Load Data** - Read parquet file from S3
3. **Preprocessing** - Convert data types and create lag features
4. **Prepare for DeepAR** - Format data as JSON for training
5. **Train Model** - Launch SageMaker training job
6. **Deploy Model** - Create real-time endpoint
7. **Generate Predictions** - Forecast 24 hours for 5 random resources
8. **Visualize Results** - Plot predictions with confidence intervals
9. **Calculate Metrics** - Evaluate model performance

### 3. Model Configuration
Key hyperparameters (adjustable in the notebook):
- `prediction_length`: 24 hours (forecast horizon)
- `context_length`: 168 hours (7 days of historical data)
- `epochs`: 100 (with early stopping)
- `learning_rate`: 0.001
- `num_layers`: 3
- `num_cells`: 40

## Model Architecture

**DeepAR** is a supervised learning algorithm for forecasting time series using recurrent neural networks (RNN). It:
- Learns across multiple related time series
- Produces probabilistic forecasts
- Handles missing values and varying time series lengths
- Incorporates categorical and dynamic features

### Features Used
- **Dynamic Features**: day_of_week, is_exam_period, is_peak_hour (change over time)
- **Static Categorical Features**: resource_type, location (constant per resource)
- **Target**: occupancy_rate

## Output

### Predictions
For each selected resource, the model generates:
- Mean forecast for next 24 hours
- 10th percentile (lower bound)
- 50th percentile (median)
- 90th percentile (upper bound)

### Visualizations
- Line plots showing actual vs predicted occupancy
- Shaded confidence intervals
- Saved as `occupancy_forecasts.png`

### Metrics
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R²** (R-squared): Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Percentage error

## Cost Considerations

### Training
- Instance: `ml.c5.2xlarge`
- Duration: ~10-30 minutes (depends on data size)
- Cost: ~$0.50-1.50 per training run

### Inference Endpoint
- Instance: `ml.m5.xlarge`
- Cost: ~$0.23/hour while deployed
- **Important**: Delete endpoint when not in use to avoid charges

### Cleanup
Uncomment and run the cleanup cell to delete the endpoint:
```python
predictor.delete_endpoint()
```

## Troubleshooting

### Common Issues

**1. Not enough data**
- Ensure each resource has at least 192 hours (8 days) of data
- Check for gaps in timestamps

**2. Training fails**
- Verify S3 permissions
- Check data format (all required columns present)
- Review CloudWatch logs (link provided in error message)

**3. Poor predictions**
- Increase `context_length` for more historical data
- Adjust `epochs` or `learning_rate`
- Add more dynamic features
- Ensure data quality (no extreme outliers)

**4. Endpoint errors**
- Verify dynamic features span full prediction range
- Check categorical feature consistency

## Extending the Model

### Add More Features
Include additional dynamic features in the data preparation:
```python
dynamic_feat.append(resource_df['new_feature_encoded'].values.tolist())
```

### Adjust Forecast Horizon
Change `prediction_length` to forecast different time periods:
```python
prediction_length = 48  # 2 days ahead
```

### Batch Predictions
For forecasting all resources, modify the prediction loop to iterate through all test samples instead of random selection.

## References

- [Amazon SageMaker DeepAR Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)
- [DeepAR Research Paper](https://arxiv.org/abs/1704.04110)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)

## License

This notebook is provided as-is for educational and demonstration purposes.

## Support

For issues or questions:
1. Check SageMaker documentation
2. Review CloudWatch logs for training/endpoint errors
3. Verify data format and S3 permissions

