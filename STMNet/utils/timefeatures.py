from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    Base class for time feature extraction
    Converts datetime indices to normalized numerical features
    """

    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract time feature from datetime index - must be implemented by subclasses"""
        pass

    def __repr__(self):
        """String representation of the feature class"""
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second of minute encoded as normalized value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract second of minute feature normalized to [-0.5, 0.5]"""
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as normalized value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract minute of hour feature normalized to [-0.5, 0.5]"""
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as normalized value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract hour of day feature normalized to [-0.5, 0.5]"""
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Day of week encoded as normalized value between [-0.5, 0.5] (Monday=0, Sunday=6)"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract day of week feature normalized to [-0.5, 0.5]"""
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as normalized value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract day of month feature normalized to [-0.5, 0.5]"""
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as normalized value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract day of year feature normalized to [-0.5, 0.5]"""
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as normalized value between [-0.5, 0.5] (January=1, December=12)"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract month of year feature normalized to [-0.5, 0.5]"""
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as normalized value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract week of year feature normalized to [-0.5, 0.5]"""
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features appropriate for the given frequency string

    Maps frequency strings to relevant time features for temporal modeling
    Different frequencies require different levels of temporal granularity

    Parameters
    ----------
    freq_str : str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    Returns
    -------
    List[TimeFeature]
        List of time feature classes appropriate for the given frequency

    Raises
    ------
    RuntimeError
        If the frequency string is not supported
    """
    # Mapping of pandas offset types to relevant time features
    features_by_offsets = {
        offsets.YearEnd: [],  # Yearly frequency - no intra-year features needed
        offsets.QuarterEnd: [MonthOfYear],  # Quarterly - month information
        offsets.MonthEnd: [MonthOfYear],  # Monthly - month information
        offsets.Week: [DayOfMonth, WeekOfYear],  # Weekly - day and week of year
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],  # Daily - day-level features
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # Business days - same as daily
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],  # Hourly - hour and day features
        offsets.Minute: [  # Minutely - detailed temporal features
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [  # Secondly - most granular temporal features
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    # Convert frequency string to pandas offset object
    offset = to_offset(freq_str)

    # Find matching offset type and return corresponding features
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # Error message for unsupported frequencies
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    """
    Extract multiple time features from datetime indices and stack them vertically

    Args:
        dates: Datetime indices to extract features from
        freq: Frequency string indicating the temporal granularity

    Returns:
        numpy.ndarray: Stacked time features of shape [n_features, n_timesteps]
    """
    # Get appropriate feature classes for the frequency
    feature_classes = time_features_from_frequency_str(freq)

    # Extract each feature and stack vertically
    return np.vstack([feat(dates) for feat in feature_classes])