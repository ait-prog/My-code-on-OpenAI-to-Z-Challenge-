import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.plot import show
import os
import openai
import json
import requests
from io import BytesIO
import warnings
from kaggle_secrets import UserSecretsClient
from sklearn.cluster import DBSCAN
import contextily as ctx
warnings.filterwarnings('ignore')

class AmazonArchaeologyAnalyzer:
    def __init__(self):
        self.df = None
        self.df_amazon = None
        self.user_secrets = UserSecretsClient()
        self.openai_api_key = self.user_secrets.get_secret("legolas_API")
        openai.api_key = self.openai_api_key
        self.geo_df = None
        
    def load_data(self):
        """Загрузка данных об археологических местах из CSV файлов"""
        print("Загрузка данных из CSV файлов...")
        try:
            # Загрузка данных
            amazon_df = pd.read_csv('/kaggle/input/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv')
            
            # Разделяем координаты на широту и долготу
            amazon_df[['latitude', 'longitude']] = amazon_df['coordinates'].str.split(' ', expand=True).astype(float)
            
            # Обработка данных
            amazon_df = amazon_df.rename(columns={
                'name': 'name'
            })
            amazon_df['description'] = "Amazon geoglyph site"
            amazon_df['source'] = 'amazon_geoglyphs'
            
            # Выбираем нужные колонки
            self.df = amazon_df[['latitude', 'longitude', 'name', 'description', 'source']]
            
            print(f"Загружено {len(self.df)} археологических мест")
            
            # Создаем GeoDataFrame
            geometry = [Point(xy) for xy in zip(self.df['longitude'], self.df['latitude'])]
            self.geo_df = gpd.GeoDataFrame(self.df, geometry=geometry, crs="EPSG:4326")
            
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return False
        return True

    def filter_amazon_region(self):
        """Фильтрация данных по региону Амазонии"""
        amazon_bounds = {
            'min_lat': -10,
            'max_lat': 5,
            'min_lon': -80,
            'max_lon': -45
        }

        self.df_amazon = self.df[
            (self.df['latitude'] >= amazon_bounds['min_lat']) &
            (self.df['latitude'] <= amazon_bounds['max_lat']) &
            (self.df['longitude'] >= amazon_bounds['min_lon']) &
            (self.df['longitude'] <= amazon_bounds['max_lon'])
        ]
        
        # Обновляем GeoDataFrame
        geometry = [Point(xy) for xy in zip(self.df_amazon['longitude'], self.df_amazon['latitude'])]
        self.geo_df = gpd.GeoDataFrame(self.df_amazon, geometry=geometry, crs="EPSG:4326")
        
        print(f"Отфильтровано {len(self.df_amazon)} записей в регионе Амазонии")

    def get_elevation_data(self, lat, lon):
        """Получение данных о высоте из SRTM"""
        # Здесь можно использовать API для получения высотных данных
        # Например, NASA SRTM или другие источники
        # Пока используем заглушку
        return {
            'elevation': np.random.normal(500, 200),
            'slope': np.random.normal(15, 5),
            'aspect': np.random.uniform(0, 360)
        }

    def prepare_features(self):
        """Подготовка признаков для анализа"""
        print("Подготовка признаков...")
        self.df_amazon['elevation_features'] = self.df_amazon.apply(
            lambda row: self.get_elevation_data(row['latitude'], row['longitude']), 
            axis=1
        )

        # Развертывание признаков
        self.df_amazon['elevation'] = self.df_amazon['elevation_features'].apply(lambda x: x['elevation'])
        self.df_amazon['slope'] = self.df_amazon['elevation_features'].apply(lambda x: x['slope'])
        self.df_amazon['aspect'] = self.df_amazon['elevation_features'].apply(lambda x: x['aspect'])

    def analyze_with_gpt4(self, location_data):
        """Анализ местоположения с помощью GPT-4"""
        try:
            prompt = f"""
            Проанализируйте следующие данные о потенциальном археологическом месте:
            Широта: {location_data['latitude']}
            Долгота: {location_data['longitude']}
            Высота: {location_data['elevation']} м
            Уклон: {location_data['slope']} градусов
            Аспект: {location_data['aspect']} градусов
            
            Оцените вероятность нахождения археологического памятника в этом месте,
            учитывая исторический контекст и особенности рельефа.
            Верните ответ в формате JSON с полями:
            - probability: число от 0 до 1
            - explanation: текстовое объяснение
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Вы - эксперт по археологии Амазонии. Отвечайте строго в формате JSON."},
                    {"role": "user", "content": prompt}
                ]
            )

            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except json.JSONDecodeError:
                print("Ошибка при парсинге JSON ответа от GPT-4")
                return None

        except Exception as e:
            print(f"Ошибка при анализе с GPT-4: {e}")
            return None

    def create_prediction_map(self):
        """Создание интерактивной карты с предсказаниями"""
        print("Создание карты предсказаний...")
        
        # Получаем предсказания для каждой точки
        predictions = []
        for _, row in self.df_amazon.iterrows():
            location_data = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'elevation': row['elevation'],
                'slope': row['slope'],
                'aspect': row['aspect']
            }
            gpt_analysis = self.analyze_with_gpt4(location_data)
            if gpt_analysis:
                predictions.append(gpt_analysis['probability'])
            else:
                predictions.append(0.0)

        # Создаем карту
        m = folium.Map(
            location=[self.df_amazon['latitude'].mean(), 
                     self.df_amazon['longitude'].mean()], 
            zoom_start=6,
            tiles='CartoDB positron'
        )

        # Добавляем тепловую карту
        heat_data = [[row['latitude'], row['longitude'], pred] 
                    for row, pred in zip(self.df_amazon.to_dict('records'), predictions)]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)

        # Добавляем маркеры для известных мест с улучшенной информацией
        marker_cluster = MarkerCluster().add_to(m)
        for idx, row in self.df_amazon.iterrows():
            # Создаем информативное всплывающее окно
            popup_text = f"""
            <b>Вероятность находки:</b> {predictions[idx]:.2f}<br>
            <b>Высота:</b> {row['elevation']:.1f} м<br>
            <b>Уклон:</b> {row['slope']:.1f}°<br>
            <b>Аспект:</b> {row['aspect']:.1f}°<br>
            """
            
            # Определяем цвет маркера на основе вероятности
            if predictions[idx] > 0.7:
                color = 'red'
            elif predictions[idx] > 0.4:
                color = 'orange'
            else:
                color = 'blue'
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon='info-sign'),
                tooltip=f"Вероятность: {predictions[idx]:.2f}"
            ).add_to(marker_cluster)

        # Добавляем слой с границами кластеров
        if 'cluster' in self.df_amazon.columns:
            for cluster_id in set(self.df_amazon['cluster']):
                if cluster_id != -1:  # Пропускаем шумовые точки
                    cluster_points = self.df_amazon[self.df_amazon['cluster'] == cluster_id]
                    # Создаем выпуклую оболочку для кластера
                    from scipy.spatial import ConvexHull
                    points = cluster_points[['latitude', 'longitude']].values
                    if len(points) >= 3:  # Нужно минимум 3 точки для выпуклой оболочки
                        hull = ConvexHull(points)
                        # Создаем полигон для кластера
                        cluster_polygon = folium.Polygon(
                            locations=points[hull.vertices],
                            color='green',
                            fill=True,
                            fill_color='green',
                            fill_opacity=0.1,
                            popup=f'Кластер {cluster_id}'
                        ).add_to(m)

        # Добавляем элементы управления
        folium.LayerControl().add_to(m)

        # Сохранение карты
        m.save('amazon_archaeology_predictions.html')
        print("Карта сохранена в файл 'amazon_archaeology_predictions.html'")

    def analyze_clusters(self):
        """Анализ кластеров археологических мест"""
        print("Анализ кластеров...")
        
        # Подготовка данных для кластеризации
        coords = self.df_amazon[['latitude', 'longitude']].values
        
        # Применяем DBSCAN с оптимизированными параметрами
        db = DBSCAN(eps=0.5, min_samples=3).fit(coords)
        self.df_amazon['cluster'] = db.labels_
        
        # Создаем фигуру с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Визуализация кластеров на карте
        scatter = ax1.scatter(
            self.df_amazon['longitude'],
            self.df_amazon['latitude'],
            c=self.df_amazon['cluster'],
            cmap='viridis',
            s=50,
            alpha=0.6
        )
        
        # Добавляем границы кластеров
        for cluster_id in set(db.labels_):
            if cluster_id != -1:  # Пропускаем шумовые точки
                cluster_points = self.df_amazon[self.df_amazon['cluster'] == cluster_id]
                points = cluster_points[['longitude', 'latitude']].values
                if len(points) >= 3:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax1.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.3)
        
        ax1.set_title('Кластеры археологических мест в Амазонии')
        ax1.set_xlabel('Долгота')
        ax1.set_ylabel('Широта')
        plt.colorbar(scatter, ax=ax1, label='Кластер')
        
        # Статистика по кластерам
        cluster_stats = []
        for cluster_id in set(db.labels_):
            if cluster_id != -1:
                cluster_data = self.df_amazon[self.df_amazon['cluster'] == cluster_id]
                stats = {
                    'Кластер': cluster_id,
                    'Количество мест': len(cluster_data),
                    'Средняя высота': cluster_data['elevation'].mean(),
                    'Средний уклон': cluster_data['slope'].mean(),
                    'Площадь (кв. км)': ConvexHull(cluster_data[['longitude', 'latitude']].values).volume
                }
                cluster_stats.append(stats)
        
        # Визуализация статистики
        if cluster_stats:
            stats_df = pd.DataFrame(cluster_stats)
            stats_df.plot(kind='bar', x='Кластер', y='Количество мест', ax=ax2)
            ax2.set_title('Статистика по кластерам')
            ax2.set_xlabel('Номер кластера')
            ax2.set_ylabel('Количество мест')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Сохраняем статистику в CSV
        if cluster_stats:
            pd.DataFrame(cluster_stats).to_csv('cluster_statistics.csv', index=False)
        
        print(f"Найдено {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)} кластеров")
        return db.labels_

    def visualize_elevation_distribution(self):
        """Визуализация распределения высот"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df_amazon, x='elevation', bins=30)
        plt.title('Распределение высот археологических мест')
        plt.xlabel('Высота (м)')
        plt.ylabel('Количество мест')
        plt.savefig('elevation_distribution.png')
        plt.close()

    def create_geospatial_plot(self):
        """Создание геопространственного графика"""
        if self.geo_df is not None:
            # Проекция для Южной Америки
            self.geo_df = self.geo_df.to_crs(epsg=3857)
            
            # Создание графика
            fig, ax = plt.subplots(figsize=(15, 15))
            self.geo_df.plot(ax=ax, alpha=0.5, color='red')
            
            # Добавление базовой карты
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            
            plt.title('Археологические места в Амазонии')
            plt.savefig('geospatial_plot.png')
            plt.close()

    def run_analysis(self):
        """Запуск полного анализа"""
        if not self.load_data():
            return
        
        self.filter_amazon_region()
        self.prepare_features()
        self.create_prediction_map()
        self.analyze_clusters()
        self.visualize_elevation_distribution()
        self.create_geospatial_plot()

        # Пример анализа конкретного места
        sample_location = self.df_amazon.iloc[0]
        gpt_analysis = self.analyze_with_gpt4(sample_location)
        if gpt_analysis:
            print("\nРезультаты анализа GPT-4:")
            print(json.dumps(gpt_analysis, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    analyzer = AmazonArchaeologyAnalyzer()
    analyzer.run_analysis()
