# AnomalyDetector

In this implementation, I combined three different anomaly detection algorithms—Holt-Winters, EMA (Exponential Moving Average) with Z-Score, and CUSUM—to leverage the strengths of each in detecting various types of anomalies. Holt-Winters is particularly effective for capturing anomalies in data with strong seasonal patterns due to its handling of both trend and seasonality. The EMA Z-Score method is well-suited for detecting anomalies based on rapid deviations from a moving average, making it useful for spotting short-term shifts or spikes. Finally, CUSUM (Cumulative Sum) is highly sensitive to small, sustained changes in the data, allowing it to detect gradual drifts or persistent shifts. By combining these algorithms, I created a robust system capable of detecting a wide range of anomaly types, from abrupt spikes to long-term trends, while maintaining sensitivity to both local and global deviations in the data stream. This multi-layered approach increases detection accuracy across different scenarios.

Make sure to install the requirements first!

```bash
pip install -r requirements.txt
```


Example Output:

![Anomalies Histogram](https://github.com/MituIustin/AnomalyDetector/blob/main/fig1.png)
