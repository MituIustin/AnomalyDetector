# In this implementation, I combined three different anomaly detection algorithms
# - Holt-Winters
# - EMA (Exponential Moving Average) with Z-Score 
# - CUSUM
# to leverage the strengths of each in detecting various types of anomalies. 
#     Holt-Winters is particularly effective for capturing anomalies in data with 
# strong seasonal patterns due to its handling of both trend and seasonality. 
#     The EMA Z-Score method is well-suited for detecting anomalies based on rapid 
# deviations from a moving average, making it useful for spotting short-term shifts 
# or spikes. 
#     Finally, CUSUM (Cumulative Sum) is highly sensitive to small, sustained changes 
# in the data, allowing it to detect gradual drifts or persistent shifts. 
#     By combining these algorithms, I created a robust system capable of detecting 
# a wide range of anomaly types, from abrupt spikes to long-term trends, while 
# maintaining sensitivity to both local and global deviations in the data stream. 
# This multi-layered approach increases detection accuracy across different scenarios.

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

class HoltWintersAnomalyDetector:
    def __init__(self, seasonal_period):
        # Initializes the Holt-Winters anomaly detector
        # seasonal_period: the period of seasonality in the data
        self.model = None
        self.seasonal_period = seasonal_period
        self.fitted = False
        self.history = []
        self.errors = []

    def update(self, value):
        # Adds the new value to the historical data
        self.history.append(value)
        
        # If there is enough data to establish seasonality, fit the model
        if len(self.history) >= 2 * self.seasonal_period:
            if not self.fitted:
                # First time fitting the Holt-Winters model
                self.model = ExponentialSmoothing(
                    self.history, trend='add', seasonal='add', seasonal_periods=self.seasonal_period
                ).fit()
                self.fitted = True
            else:
                # Refit the model with new data points as they come in
                self.model = ExponentialSmoothing(
                    self.history, trend='add', seasonal='add', seasonal_periods=self.seasonal_period
                ).fit()
                
    def detect(self, value, threshold=3.0):
        # If the model is fitted, make a prediction
        if self.fitted:
            forecast = self.model.forecast(steps=1)[0]  # Predict the next value
            error = abs(value - forecast)  # Calculate the absolute error between real value and forecast
            self.errors.append(error)
            
            # If we have historical errors, calculate dynamic thresholds based on standard deviation
            if len(self.errors) > 1:
                mean_error = np.mean(self.errors)
                std_error = np.std(self.errors)
                dynamic_threshold = mean_error + threshold * std_error
                
                # If the error exceeds the dynamic threshold, it's an anomaly
                if error > dynamic_threshold:
                    return True
        return False

    
class EMAZScoreAnomalyDetector:
    def __init__(self, alpha=0.3, z_threshold=2.0):
        # Initialize the EMA (Exponential Moving Average) Z-Score detector
        self.ema = None
        self.alpha = alpha  # Smoothing factor for the EMA
        self.z_threshold = z_threshold  # Threshold for detecting an anomaly based on Z-score
        self.mean = None
        self.std_dev = None

    def update_ema(self, value):
        # Updates the EMA with the new value
        if self.ema is None:
            self.ema = value  # First EMA is simply the first value
        else:
            # Recursively update the EMA
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return self.ema

    def detect(self, value):
        # Get the updated EMA value
        ema_value = self.update_ema(value)
        
        # Initialize mean and standard deviation on first pass
        if self.mean is None:
            self.mean = ema_value
            self.std_dev = 1e-6  # Small non-zero value to avoid division by zero
        else:
            # Calculate Z-score based on the deviation from the mean
            z_score = (value - self.mean) / self.std_dev
            # If Z-score exceeds the threshold, it is considered an anomaly
            if abs(z_score) > self.z_threshold:
                return True
            
            # Update mean and standard deviation based on EMA
            self.mean = self.alpha * value + (1 - self.alpha) * self.mean
            self.std_dev = self.alpha * abs(value - self.mean) + (1 - self.alpha) * self.std_dev
        
        return False

    
class CUSUMDetector:
    def __init__(self, threshold=3, drift=0.02):
        # Initialize CUSUM (Cumulative Sum) detector for anomaly detection
        self.positive_sum = 0
        self.negative_sum = 0
        self.threshold = threshold  # Threshold for detection sensitivity
        self.drift = drift  # Drift tolerance for mean adjustments
        self.mean = None

    def detect(self, value):
        if self.mean is None:
            self.mean = value  # Initialize mean on the first data point

        # Compute the deviation from the mean with drift adjustment
        diff = value - self.mean - self.drift
        self.positive_sum = max(0, self.positive_sum + diff)  # Update positive CUSUM
        self.negative_sum = max(0, self.negative_sum - diff)  # Update negative CUSUM
        
        # If either positive or negative CUSUM exceeds the threshold, it's an anomaly
        if self.positive_sum > self.threshold or self.negative_sum > self.threshold:
            return True
        
        # Adjust the mean for subsequent values
        self.mean = self.mean + self.drift * (value - self.mean)
        
        return False

    
class DataStreamSimulator:
    def __init__(self, trend_slope=0.01, seasonality_amplitude=1, noise_std=0.2, seasonal_period=50, anomaly_start_step=None, anomaly_duration=10):
        # Initialize a simulated data stream with trend, seasonality, noise, and potential anomalies
        self.trend_slope = trend_slope  # Linear trend factor
        self.seasonality_amplitude = seasonality_amplitude  # Amplitude of the seasonal component
        self.noise_std = noise_std  # Standard deviation of the noise component
        self.seasonal_period = seasonal_period  # Seasonal cycle length
        self.anomaly_start_step = anomaly_start_step  # Step at which anomalies should start
        self.anomaly_duration = anomaly_duration  # Duration of anomalies
        self.current_step = 0  # Current time step in the simulation
        self.anomaly_active = False  # Flag indicating if an anomaly is active

    def generate_next_value(self):
        # Generate the trend component
        trend = self.trend_slope * self.current_step
        
        # Generate the seasonality component as a sinusoidal function
        seasonality = self.seasonality_amplitude * np.sin(2 * np.pi * self.current_step / self.seasonal_period)
        
        # Generate random noise
        noise = np.random.normal(0, self.noise_std)
        
        # Check if the current step falls within the anomaly range
        if self.anomaly_start_step is not None and self.anomaly_start_step <= self.current_step < self.anomaly_start_step + self.anomaly_duration:
            # Modify seasonality to simulate an anomaly (e.g., increase or decrease in magnitude)
            seasonality += np.random.uniform(3, 6)
            self.anomaly_active = True
        else:
            self.anomaly_active = False

        # The final value is the sum of trend, seasonality, and noise
        value = trend + seasonality + noise
        
        # Increment the time step for the next value generation
        self.current_step += 1
        
        return value, self.anomaly_active


def monitor_data_stream(simulator, steps=200):
    # Function to monitor the simulated data stream and detect anomalies
    values = []
    anomalies_ema_z = []
    anomalies_cusum = []
    anomalies_hw = []  
    
    for i in range(steps):
        # Generate the next value in the data stream
        value, is_generated_anomaly = simulator.generate_next_value()
        values.append(value)
        anomalies_hw.append(is_generated_anomaly)
        
        # Update anomaly detectors with the new value
        holt_winters_detector.update(value)
        is_anomaly_hw = holt_winters_detector.detect(value)
        
        is_anomaly_ema_z = ema_zscore_detector.detect(value)
        is_anomaly_cusum = cusum_detector.detect(value)
        
        # Track the detected anomalies
        anomalies_ema_z.append(is_anomaly_ema_z)
        anomalies_cusum.append(is_anomaly_cusum)
    
    # Plot the data stream along with the detected anomalies
    plt.figure(figsize=(12, 6))
    plt.plot(values, label="Data Stream", color="blue")
    
    # Plot detected EMA + Z-Score anomalies
    plt.scatter(
        np.arange(steps)[anomalies_ema_z], 
        np.array(values)[anomalies_ema_z], 
        color="orange", label="EMA Z-Score Anomaly", marker="o"
    )
    
    # Plot detected CUSUM anomalies
    plt.scatter(
        np.arange(steps)[anomalies_cusum], 
        np.array(values)[anomalies_cusum], 
        color="green", label="CUSUM Anomaly", marker="^"
    )
    
    # Plot generated seasonal anomalies
    plt.scatter(
        np.arange(steps)[anomalies_hw], 
        np.array(values)[anomalies_hw], 
        color="purple", label="Generated Seasonal Anomaly", marker="s"
    )
    
    plt.title("Data Stream with Detected Anomalies")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# Initialize detectors
holt_winters_detector = HoltWintersAnomalyDetector(seasonal_period=50)
ema_zscore_detector = EMAZScoreAnomalyDetector(alpha=0.3, z_threshold=3)
cusum_detector = CUSUMDetector(threshold=5, drift=0.02)

# Set up the simulator with trend, seasonality, noise, and anomalies
simulator = DataStreamSimulator(
    trend_slope=0.01, 
    seasonality_amplitude=2, 
    noise_std=0.5, 
    seasonal_period=50, 
    anomaly_start_step=150,  # Start step for anomalies
    anomaly_duration=20      # Duration of anomalies
)

# Run the detectors on the simulated data stream
monitor_data_stream(simulator, steps=400)
