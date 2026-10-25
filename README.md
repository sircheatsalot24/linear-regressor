# Split Linear Regressor

`SplitLinearRegressor()` is a Linear Regressor machine learning "model" that uses a splitter method to find a line of best fit over a dataset of `x` and `y` points.

## To Use

To use the regressor, just `uv sync`, create a `SplitLinearRegressor()` object in a python file, use `.train(points)` with all your training points (a sample is included in `sample_points.py`), and `.predict(x)` your `x` value(s) to recieve an estimated `y`. To visually see all your points and the line, use `.show_plot()` to pull up a `matplotlib` generated graph.

## Method

We first calculate the midpoint of the data (`(max(xs)-min(xs))/2`) and split the data on that midpoint, creating a left and right side. From there, we take the average of all `x` and `y` values on both sides and combine them to form two "average" points on each side. Then, we calculate the slope of the line, using the slope formula, `m = (y_2 - y_1) / (x_2 - x_1)`. Finally, we use point slope form of a linear equation, `y-y_1 = m(x-x_1)`, solving for y, we get `y = mx-mx_1 + y_1`, at which point we can plug in `m, x_1`, and `y_1`, getting both the slope and y-intercept of the line of best fit.

## Evaluation

In practice, the model beat Ordinary Least Squares (the most popular linear regression method) abour 30% of the time over a test of ~7000 datasets, but trailed behind about half a percent behind OLS the rest of the time. The upside to this model is the lack of overhead, considerably less than OLS. Compared using mean average error (MAE).

## Next Steps

Next possible steps include using median instead of mean for the two points to make the line, because median isn't affected by outliers (mean can skew), using 3 different splits and ignoring the middle to get more accurate at the endpoints, etc.

**Not vibe coded.**
