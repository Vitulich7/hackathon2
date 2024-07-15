import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy import stats
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import cluster
from sklearn import feature_selection

taxi_data = pd.read_csv("data/train.csv")
print('Train data shape: {}'.format(taxi_data.shape))
taxi_data.head()



#Первичная обработка данных



# Переводим признак pickup_datetime в тип данных datetime
taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

# Определяем временные рамки (без учета времени)
start_date = taxi_data['pickup_datetime'].dt.date.min()
end_date = taxi_data['pickup_datetime'].dt.date.max()

print(f"Временные рамки данных: {start_date} - {end_date}")

# Подсчитываем общее количество пропущенных значений
total_missing = taxi_data.isnull().sum().sum()

print(f"Общее количество пропущенных значений: {total_missing}")

# а) Количество уникальных таксопарков
num_unique_vendors = taxi_data['vendor_id'].nunique()
print(f"Количество уникальных таксопарков: {num_unique_vendors}")

# б) Максимальное количество пассажиров
max_passengers = taxi_data['passenger_count'].max()
print(f"Максимальное количество пассажиров: {max_passengers}")

# в) Средняя и медианная длительность поездки в секундах
trip_duration_sec = taxi_data['trip_duration'].astype(int)
mean_trip_duration = trip_duration_sec.mean().round(0)
median_trip_duration = trip_duration_sec.median().round(0)
print(f"Средняя длительность поездки: {int(mean_trip_duration)} секунд")
print(f"Медианная длительность поездки: {int(median_trip_duration)} секунд")

# г) Минимальное и максимальное время поездки в секундах
min_trip_duration = trip_duration_sec.min()
max_trip_duration = trip_duration_sec.max()
print(f"Минимальное время поездки: {min_trip_duration} секунд")
print(f"Максимальное время поездки: {max_trip_duration} секунд")


def add_datetime_features(df):
    # Добавляем столбец pickup_date
    df['pickup_date'] = df['pickup_datetime'].dt.date

    # Добавляем столбец pickup_hour
    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    # Добавляем столбец pickup_day_of_week
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.day_of_week

    return df


# Применяем функцию к исходным данным
taxi_data = add_datetime_features(taxi_data)

# а) Количество поездок в субботу
saturday_trips = taxi_data[taxi_data['pickup_day_of_week'] == 5].shape[0]
print(f"Количество поездок в субботу: {saturday_trips}")

# б) Среднее количество поездок в день
total_trips = taxi_data.shape[0]
num_days = (taxi_data['pickup_date'].max() - taxi_data['pickup_date'].min()).days + 1
average_daily_trips = round(total_trips / num_days)
print(f"Среднее количество поездок в день: {average_daily_trips}")


def add_holiday_features(trips_df, holidays_df):
    # Объединяем таблицы по дате
    merged_df = pd.merge(trips_df, holidays_df, left_on='pickup_date', right_on='holiday_date', how='left')

    # Заполняем пропущенные значения 0, чтобы обозначить, что это не праздничный день
    merged_df['is_holiday'] = merged_df['is_holiday'].fillna(0)

    # Создаем новый столбец pickup_holiday
    merged_df['pickup_holiday'] = merged_df['is_holiday'].astype(int)

    return merged_df


# Применяем функцию к исходным данным
taxi_data = add_holiday_features(taxi_data, holiday_data)

# Вычисляем медианную длительность поездки в праздничные дни
holiday_trip_duration = taxi_data[taxi_data['pickup_holiday'] == 1]['trip_duration'].astype(int)
median_holiday_trip_duration = holiday_trip_duration.median().round(0)

print(f"Медианная длительность поездки в праздничные дни: {int(median_holiday_trip_duration)} секунд")


def add_osrm_features(trips_df, osrm_df):
    # Объединяем таблицы по координатам
    merged_df = pd.merge(trips_df, osrm_df,
                         left_on=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'],
                         right_on=['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude'], how='left')

    # Добавляем столбцы с информацией из OSRM
    merged_df['total_distance'] = merged_df['distance']
    merged_df['total_travel_time'] = merged_df['duration']
    merged_df['number_of_steps'] = merged_df['number_of_steps']

    return merged_df


# Применяем функцию к исходным данным
taxi_data = add_osrm_features(taxi_data, osrm_data)

# а) Разница между медианной длительностью поездки в данных и медианной длительностью поездки из OSRM
trip_duration_sec = taxi_data['trip_duration'].astype(int)
osrm_trip_duration_sec = taxi_data['total_travel_time'].astype(int)

median_trip_duration = trip_duration_sec.median()
median_osrm_trip_duration = osrm_trip_duration_sec.median()

diff_median_duration = median_trip_duration - median_osrm_trip_duration
print(
    f"Разница между медианной длительностью поездки в данных и медианной длительностью поездки из OSRM: {int(diff_median_duration)} секунд")

# б) Количество пропусков в столбцах с информацией из OSRM API
num_missing_total_distance = taxi_data['total_distance'].isna().sum()
num_missing_total_travel_time = taxi_data['total_travel_time'].isna().sum()
num_missing_number_of_steps = taxi_data['number_of_steps'].isna().sum()

print(f"Количество пропусков в столбце total_distance: {num_missing_total_distance}")
print(f"Количество пропусков в столбце total_travel_time: {num_missing_total_travel_time}")
print(f"Количество пропусков в столбце number_of_steps: {num_missing_number_of_steps}")

import numpy as np


def get_haversine_distance(lat1, lng1, lat2, lng2):
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # радиус земли в километрах
    EARTH_RADIUS = 6371
    # считаем кратчайшее расстояние h по формуле Хаверсина
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def get_angle_direction(lat1, lng1, lat2, lng2):
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # считаем угол направления движения alpha по формуле угла пеленга
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha


def add_geographical_features(trips_df):
    # Вычисляем расстояние Хаверсина
    trips_df['haversine_distance'] = trips_df.apply(
        lambda row: get_haversine_distance(row['pickup_latitude'], row['pickup_longitude'],
                                           row['dropoff_latitude'], row['dropoff_longitude']),
        axis=1)

    # Вычисляем направление движения
    trips_df['direction'] = trips_df.apply(
        lambda row: get_angle_direction(row['pickup_latitude'], row['pickup_longitude'],
                                        row['dropoff_latitude'], row['dropoff_longitude']),
        axis=1)

    return trips_df


# Применяем функцию к исходным данным
taxi_data = add_geographical_features(taxi_data)

# Вычисляем медианное расстояние Хаверсина
median_haversine_distance = taxi_data['haversine_distance'].median().round(2)
print(f"Медианное расстояние Хаверсина поездок: {median_haversine_distance} км")

from sklearn import cluster


def add_cluster_features(trips_df, kmeans_model):
    # Объединяем координаты начала и конца поездки в один массив
    coords = np.hstack((trips_df[['pickup_latitude', 'pickup_longitude']],
                        trips_df[['dropoff_latitude', 'dropoff_longitude']]))

    # Предсказываем кластер для каждой точки
    trips_df['geo_cluster'] = kmeans_model.predict(coords)

    return trips_df


# Применяем функцию к исходным данным
taxi_data = add_cluster_features(taxi_data, kmeans)

# Находим размер наименьшего кластера
cluster_sizes = taxi_data['geo_cluster'].value_counts()
smallest_cluster_size = cluster_sizes.min()

print(f"Количество поездок в наименьшем географическом кластере: {smallest_cluster_size}")

import pandas as pd


def add_weather_features(trips_df, weather_df):
    # Объединяем таблицы по дате и времени
    merged_df = pd.merge(trips_df, weather_df, left_on=['pickup_datetime'], right_on=['datetime'], how='left')

    # Добавляем столбцы с погодными условиями
    merged_df['temperature'] = merged_df['temperature']
    merged_df['visibility'] = merged_df['visibility']
    merged_df['wind_speed'] = merged_df['wind_speed']
    merged_df['precip'] = merged_df['precip']
    merged_df['events'] = merged_df['events']

    return merged_df


# Применяем функцию к исходным данным
taxi_data = add_weather_features(taxi_data, weather_data)

# а) Количество поездок в снежную погоду
snow_trips = taxi_data[taxi_data['events'].str.contains('Snow')].shape[0]
print(f"Количество поездок в снежную погоду: {snow_trips}")

# б) Процент пропусков в столбцах с погодными условиями
total_rows = taxi_data.shape[0]
missing_rows = taxi_data[['temperature', 'visibility', 'wind_speed', 'precip', 'events']].isna().any(axis=1).sum()
missing_percent = (missing_rows / total_rows) * 100
print(f"Процент пропусков в столбцах с погодными условиями: {missing_percent:.2f}%")

import pandas as pd


def fill_null_weather_data(trips_df):
    # Заполнение пропусков в столбцах с погодными условиями
    trips_df['pickup_date'] = pd.to_datetime(trips_df['pickup_datetime']).dt.date

    # Заполнение пропусков в temperature, visibility, wind_speed, precip
    for col in ['temperature', 'visibility', 'wind_speed', 'precip']:
        trips_df[col] = trips_df.groupby('pickup_date')[col].transform(lambda x: x.fillna(x.median()))

    # Заполнение пропусков в events
    trips_df['events'] = trips_df['events'].fillna('None')

    # Заполнение пропусков в столбцах из OSRM API
    for col in ['total_distance', 'total_travel_time', 'number_of_steps']:
        trips_df[col] = trips_df[col].fillna(trips_df[col].median())

    return trips_df


# Применяем функцию к исходным данным
taxi_data = fill_null_weather_data(taxi_data)

# Находим медиану в столбце temperature после заполнения пропусков
temperature_median = taxi_data['temperature'].median()
print(f"Медиана в столбце temperature после заполнения пропусков: {temperature_median:.1f}")


avg_speed = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x=avg_speed.index, y=avg_speed, ax=ax)
ax.set_xlabel('Index')
ax.set_ylabel('Average speed');

import pandas as pd


def remove_outliers(trips_df):
    # а) Удаление поездок длительностью более 24 часов
    long_trips = trips_df[trips_df['total_travel_time'] > 24 * 3600]
    num_long_trips = long_trips.shape[0]
    trips_df = trips_df[trips_df['total_travel_time'] <= 24 * 3600]

    # б) Удаление поездок со скоростью более 300 км/ч
    high_speed_trips = trips_df[trips_df['total_distance'] / trips_df['total_travel_time'] > 300 / 3.6]
    num_high_speed_trips = high_speed_trips.shape[0]
    trips_df = trips_df[trips_df['total_distance'] / trips_df['total_travel_time'] <= 300 / 3.6]

    return trips_df, num_long_trips, num_high_speed_trips


# Применяем функцию к исходным данным
taxi_data, num_long_trips, num_high_speed_trips = remove_outliers(taxi_data)

print(f"Количество выбросов по признаку длительности поездки: {num_long_trips}")
print(f"Количество выбросов по признаку скорости: {num_high_speed_trips}")



#Разведывательный анализ данных (EDA)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dagostino

# Логарифмируем целевой признак
taxi_data['trip_duration_log'] = np.log(taxi_data['trip_duration'] + 1)

# Построение гистограммы и коробчатой диаграммы
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
taxi_data['trip_duration_log'].hist(bins=30)
plt.title('Гистограмма длительности поездок (log)')
plt.xlabel('Длительность поездки (log)')
plt.ylabel('Количество')

plt.subplot(1, 2, 2)
taxi_data['trip_duration_log'].plot(kind='box')
plt.title('Коробчатая диаграмма длительности поездок (log)')
plt.xlabel('Длительность поездки (log)')
plt.show()

# Тест Д'Агостино на нормальность распределения
stat, p_value = dagostino(taxi_data['trip_duration_log'])
print(f"Статистика теста Д'Агостино: {stat:.2f}")
print(f"p-value: {p_value:.2f}")

# Определение нормальности распределения
alpha = 0.05
if p_value > alpha:
    print("Распределение длительности поездок в логарифмическом масштабе является нормальным.")
else:
    print("Распределение длительности поездок в логарифмическом масштабе не является нормальным.")


import matplotlib.pyplot as plt

# Построение наложенных гистограмм
plt.figure(figsize=(12, 6))
taxi_data.groupby('vendor_id')['trip_duration_log'].hist(bins=30, alpha=0.5, density=True)
plt.legend(['Vendor 1', 'Vendor 2'])
plt.title('Распределение длительности поездок (log) по таксопаркам')
plt.xlabel('Длительность поездки (log)')
plt.ylabel('Плотность')
plt.show()


import matplotlib.pyplot as plt

# Построение наложенных гистограмм
plt.figure(figsize=(12, 6))
taxi_data.groupby('store_and_fwd_flag')['trip_duration_log'].hist(bins=30, alpha=0.5, density=True)
plt.legend(['Без отправки сообщения', 'С отправкой сообщения'])
plt.title('Распределение длительности поездок (log) по отправке сообщения поставщику')
plt.xlabel('Длительность поездки (log)')
plt.ylabel('Плотность')
plt.show()


import matplotlib.pyplot as plt
import pandas as pd

# Создаем новый признак "hour" из столбца "pickup_datetime"
taxi_data['hour'] = taxi_data['pickup_datetime'].dt.hour

# Группируем данные по часу и считаем количество поездок
trip_count_by_hour = taxi_data.groupby('hour')['trip_duration'].count()

# Построение графика
plt.figure(figsize=(12, 6))
trip_count_by_hour.plot(kind='bar')
plt.title('Распределение количества поездок по часам дня')
plt.xlabel('Час')
plt.ylabel('Количество поездок')
plt.show()


import matplotlib.pyplot as plt
import pandas as pd

# Создаем новый признак "day_of_week" из столбца "pickup_datetime"
taxi_data['day_of_week'] = taxi_data['pickup_datetime'].dt.day_name()

# Группируем данные по дню недели и считаем количество поездок
trip_count_by_day = taxi_data.groupby('day_of_week')['trip_duration'].count()

# Построение графика
plt.figure(figsize=(12, 6))
trip_count_by_day.plot(kind='bar')
plt.title('Распределение количества поездок по дням недели')
plt.xlabel('День недели')
plt.ylabel('Количество поездок')
plt.show()


import matplotlib.pyplot as plt
import pandas as pd

# Группируем данные по дню недели и вычисляем медианную длительность поездки
median_trip_duration_by_day = taxi_data.groupby('day_of_week')['trip_duration'].median()

# Построение графика
plt.figure(figsize=(12, 6))
median_trip_duration_by_day.plot(kind='bar')
plt.title('Медианная длительность поездок по дням недели')
plt.xlabel('День недели')
plt.ylabel('Медианная длительность, сек')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Создаем новые признаки "pickup_hour" и "pickup_day_of_week" из столбца "pickup_datetime"
taxi_data['pickup_hour'] = taxi_data['pickup_datetime'].dt.hour
taxi_data['pickup_day_of_week'] = taxi_data['pickup_datetime'].dt.day_name()

# Создаем сводную таблицу с медианной длительностью поездки
pivot_table = taxi_data.pivot_table(index='pickup_hour', columns='pickup_day_of_week', values='trip_duration', aggfunc='median')

# Построение тепловой карты
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.0f')
plt.title('Медианная длительность поездок по часам и дням недели')
plt.xlabel('День недели')
plt.ylabel('Час')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Задаем границы для Нью-Йорка
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)

# Фильтруем данные по границам Нью-Йорка
taxi_data = taxi_data[(taxi_data['pickup_longitude'] >= city_long_border[0]) &
                     (taxi_data['pickup_longitude'] <= city_long_border[1]) &
                     (taxi_data['pickup_latitude'] >= city_lat_border[0]) &
                     (taxi_data['pickup_latitude'] <= city_lat_border[1])]

# Построение диаграммы рассеяния
plt.figure(figsize=(12, 8))
sns.scatterplot(x='pickup_longitude', y='pickup_latitude', data=taxi_data, hue='geo_cluster', s=2)
plt.title('Географическое расположение точек начала поездок')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.xlim(city_long_border)
plt.ylim(city_lat_border)
plt.show()


# Фильтруем данные по границам Нью-Йорка
taxi_data = taxi_data[(taxi_data['dropoff_longitude'] >= city_long_border[0]) &
                     (taxi_data['dropoff_longitude'] <= city_long_border[1]) &
                     (taxi_data['dropoff_latitude'] >= city_lat_border[0]) &
                     (taxi_data['dropoff_latitude'] <= city_lat_border[1])]

# Построение диаграммы рассеяния
plt.figure(figsize=(12, 8))
sns.scatterplot(x='dropoff_longitude', y='dropoff_latitude', data=taxi_data, hue='geo_cluster', s=2)
plt.title('Географическое расположение точек завершения поездок')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.xlim(city_long_border)
plt.ylim(city_lat_border)
plt.show()


print('Shape of data: {}'.format(taxi_data.shape))
print('Columns: {}'.format(taxi_data.columns))

train_data = taxi_data.copy()
train_data.head()

# a) Определение уникального признака
print('Unique values in "trip_id":', train_data['trip_id'].nunique())

# b) Определение понятия "утечка данных"
print("Утечка данных (data leak) - это ситуация, когда информация, недоступная в реальном мире, каким-то образом попадает в обучающую выборку, что приводит к завышенной оценке качества модели.")

# c) Определение признака, создающего утечку данных
print('Наличие признака "trip_id" в обучающем наборе данных создает утечку данных, так как он является уникальным идентификатором каждой поездки и не несет полезной информации для предсказания продолжительности поездки.')

# d) Исключение ненужных признаков
drop_columns = ['trip_id', 'pickup_datetime', 'pickup_date']
train_data = train_data.drop(drop_columns, axis=1)
print('Shape of data:', train_data.shape)

drop_columns = ['pickup_datetime', 'pickup_date']
train_data = train_data.drop(drop_columns, axis=1)
print('Shape of data:  {}'.format(train_data.shape))

# Кодирование признака vendor_id
train_data['vendor_id_coded'] = (train_data['vendor_id'] != 1).astype(int)

# Кодирование признака store_and_fwd_flag
train_data['store_and_fwd_flag_coded'] = (train_data['store_and_fwd_flag'] != 'N').astype(int)

# a) Расчет среднего по закодированному столбцу vendor_id
vendor_id_mean = train_data['vendor_id_coded'].mean()
print(f"Среднее по закодированному столбцу vendor_id: {vendor_id_mean:.2f}")

# б) Расчет среднего по закодированному столбцу store_and_fwd_flag
store_and_fwd_flag_mean = train_data['store_and_fwd_flag_coded'].mean()
print(f"Среднее по закодированному столбцу store_and_fwd_flag: {store_and_fwd_flag_mean:.3f}")


from sklearn.preprocessing import OneHotEncoder

# Создаем объект OneHotEncoder
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')

# Получаем закодированные признаки
data_onehot = one_hot_encoder.fit_transform(train_data[['pickup_day_of_week', 'geo_cluster', 'events']])

# Получаем имена закодированных столбцов
column_names = one_hot_encoder.get_feature_names_out(['pickup_day_of_week', 'geo_cluster', 'events'])

# Создаем DataFrame из закодированных признаков
data_onehot = pd.DataFrame(data_onehot.toarray(), columns=column_names)

# Выводим размерность DataFrame
print(f"Количество бинарных столбцов: {data_onehot.shape[1]}")


train_data = pd.concat(
    [train_data.reset_index(drop=True).drop(columns_to_change, axis=1), data_onehot],
    axis=1
)
print('Shape of data: {}'.format(train_data.shape))


X = train_data.drop(['trip_duration', 'trip_duration_log'], axis=1)
y = train_data['trip_duration']
y_log = train_data['trip_duration_log']


X_train, X_valid, y_train_log, y_valid_log = model_selection.train_test_split(
    X, y_log,
    test_size=0.33,
    random_state=42
)


from sklearn.feature_selection import SelectKBest, f_regression

# Выделим целевую переменную в отдельный массив
y = np.log1p(train_data['duration'])

# Создаем объект SelectKBest и отбираем 25 лучших признаков
selector = SelectKBest(score_func=f_regression, k=25)
X_new = selector.fit_transform(train_data.drop('duration', axis=1), y)

# Получаем названия отобранных признаков
selected_features = train_data.drop('duration', axis=1).columns[selector.get_support()]

print("Отобранные признаки:")
print(", ".join(selected_features))


from sklearn.preprocessing import MinMaxScaler

# Создаем объект MinMaxScaler
scaler = MinMaxScaler()

# Обучаем нормализатор на обучающей выборке
X_train_scaled = scaler.fit_transform(X_train)

# Применяем нормализацию к обучающей и валидационной выборкам
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Рассчитываем среднее арифметическое для первого предиктора в валидационной выборке
first_predictor_mean = X_val_scaled[:, 0].mean()
print(f"Среднее арифметическое для первого предиктора в валидационной выборке: {first_predictor_mean:.2f}")




