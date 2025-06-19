To interpret the output (`Winners` and `Costs`) and calculate the requested metrics (chip range, profit, average cost, concentration, and major holders), you can follow these steps:

### 1. **90% Chip Price Range**  
   - Use the `Costs` array to estimate the price range that covers 90% of the chips. For each stock and time step, you'll need to check the cumulative chip distribution and find the minimum and maximum prices that correspond to the 90% chip range.

   Here's the code for extracting the 90% chip range:

   ```python
   def calculate_chip_range(costs, price_levels, percentile=90):
       """
       Calculate the price range for the given percentile of chips.
       """
       chip_range = []
       for stock_id in range(costs.shape[0]):
           for t in range(costs.shape[1]):
               low_index = int(np.percentile(np.cumsum(costs[stock_id, :, t]), (100 - percentile)))
               high_index = int(np.percentile(np.cumsum(costs[stock_id, :, t]), percentile))
               chip_range.append([price_levels[low_index], price_levels[high_index]])
       return chip_range
   ```

### 2. **Profit from Closing Price (Winners)**  
   - The `Winners` array represents the percentage of chips below the closing price. You can interpret this value as the profit ratio, meaning how much of the stock is in profit. For example, a value of `0.5` indicates that 50% of the chips are profitable.

   ```python
   def calculate_profit_ratio(winners):
       """
       Calculate the profit ratio based on the winners' percentage.
       """
       return np.mean(winners > 0, axis=1)
   ```

### 3. **Average Cost**  
   - The average cost can be derived from the cumulative chip distribution and the price levels. It represents the weighted average price of the chips.

   ```python
   def calculate_average_cost(chip_dist, price_levels):
       """
       Calculate the average cost based on the chip distribution and price levels.
       """
       total_chips = np.sum(chip_dist, axis=1)
       avg_cost = np.sum(chip_dist * price_levels[:, np.newaxis], axis=1) / total_chips
       return avg_cost
   ```

### 4. **Stock Concentration**  
   - The stock concentration reflects how tightly distributed the chips are within a certain price range. The more concentrated the chips, the lower the concentration score.

   ```python
   def calculate_concentration(chip_dist):
       """
       Calculate stock concentration as the inverse of the spread of the chip distribution.
       """
       return 1 / np.var(chip_dist, axis=1)
   ```

### 5. **Identifying Major Players (Main Chips)**  
   - Major players can be defined based on various rules, such as chips not being sold when profit exceeds 30%, or chips being held during horizontal price movements.

   ```python
   def identify_major_players(chip_dist, winners, threshold=0.3):
       """
       Identify major players based on the threshold for holding chips without selling.
       """
       major_holders = (winners > threshold).astype(int)
       return np.mean(major_holders, axis=1)
   ```

### Example Output as DataFrame

Here's how to combine these metrics into a DataFrame for each stock:

```python
import pandas as pd

def generate_summary_df(chip_range, profit_ratios, avg_costs, concentrations, major_players):
    df = pd.DataFrame({
        'Chip Range Min': [r[0] for r in chip_range],
        'Chip Range Max': [r[1] for r in chip_range],
        'Profit Ratio': profit_ratios,
        'Average Cost': avg_costs,
        'Concentration': concentrations,
        'Major Players': major_players
    })
    return df

# Example usage
chip_range = calculate_chip_range(costs, price_levels)
profit_ratios = calculate_profit_ratio(winners)
avg_costs = calculate_average_cost(chip_distribution, price_levels)
concentrations = calculate_concentration(chip_distribution)
major_players = identify_major_players(chip_distribution, winners)

summary_df = generate_summary_df(chip_range, profit_ratios, avg_costs, concentrations, major_players)
print(summary_df)
```

This DataFrame will include:
- `Chip Range Min` and `Chip Range Max`: The price range covering 90% of the chips.
- `Profit Ratio`: The percentage of chips that are in profit.
- `Average Cost`: The weighted average price of the chips.
- `Concentration`: A measure of how tightly the chips are distributed.
- `Major Players`: The proportion of chips held by major players.