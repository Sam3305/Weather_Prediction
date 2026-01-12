import argparse
import os

import numpy as np
import pandas as pd


def adjust_precipitation(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()

    # Ensure required columns exist
    required = {'date', 'temp', 'humidity', 'wind_speed', 'pressure', 'precip'}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # 1) Clip to plausible daily range
    out['precip'] = out['precip'].clip(lower=0.0, upper=80.0)

    # 2) Replace hard zeros with small realistic amounts based on conditions
    #    - Higher humidity and lower pressure => higher chance and intensity
    zeros = out['precip'] <= 0.0

    # Normalize drivers into [0, 1] ranges for simple weighting
    # Humidity ~ [0,100], Pressure ~ [980, 1030] typical; use robust clipping
    h = (out['humidity'].clip(0, 100) / 100.0)
    p = 1.0 - (out['pressure'].clip(980, 1030) - 980.0) / 50.0
    # lower pressure -> higher p

    rain_likelihood = (0.6 * h + 0.4 * p).clip(0.0, 1.0)

    # Sample a small drizzle amount for most zeros; occasional moderate rain
    # Mixture: with prob=rain_likelihood*0.7 => drizzle [0.1, 1.5] mm
    #          with prob=rain_likelihood*0.2 => light rain [1.5, 7] mm
    #          else keep near dry with trace [0.0, 0.1] (avoid exact zeros)
    u = rng.random(len(out))

    drizzle = (u < (rain_likelihood * 0.7))
    light = (u >= (rain_likelihood * 0.7)) & (u < (rain_likelihood * 0.9))
    trace = ~(drizzle | light)

    new_precip = out['precip'].to_numpy(copy=True)

    # Only modify where zeros were present
    z_idx = np.where(zeros)[0]
    if len(z_idx) > 0:
        # Drizzle
        dz_idx = z_idx[drizzle[z_idx]]
        new_precip[dz_idx] = rng.uniform(0.1, 1.5, size=len(dz_idx))
        # Light rain
        lt_idx = z_idx[light[z_idx]]
        new_precip[lt_idx] = rng.uniform(1.5, 7.0, size=len(lt_idx))
        # Trace amounts
        tr_idx = z_idx[trace[z_idx]]
        new_precip[tr_idx] = rng.uniform(0.01, 0.1, size=len(tr_idx))

    out['precip'] = new_precip

    # 3) Soften extreme outliers with humidity/pressure-aware cap
    #    Higher humidity/low pressure allow higher precip caps
    dynamic_cap = 10.0 + 40.0 * rain_likelihood  # 10 to 50 mm/day
    out['precip'] = np.minimum(out['precip'], dynamic_cap)

    # Round to 2 decimals for readability
    out['precip'] = out['precip'].round(2)

    return out


def main():
    parser = argparse.ArgumentParser(
        description='Clean synthetic weather dataset.'
    )
    parser.add_argument(
        '--input', default='synthetic_weather.csv', help='Path to input CSV'
    )
    parser.add_argument(
        '--output',
        default='synthetic_weather_clean.csv',
        help='Path to output CSV',
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    cleaned = adjust_precipitation(df, seed=args.seed)
    cleaned.to_csv(args.output, index=False)
    print(
        f"Wrote cleaned dataset to {args.output}"
    )


if __name__ == '__main__':
    main()
