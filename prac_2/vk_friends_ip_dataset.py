#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Датасет IP-адресов друзей ВКонтакте и анализ центральности сети
Задача №2: Сбор информации о друзьях и друзьях друзей из ВК для членов группы
"""

import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import ipaddress
import json

class VKFriendsIPDataset:
    """Класс для создания и анализа датасета IP-адресов друзей ВКонтакте"""
    
    def __init__(self, group_size: int = 20):
        """
        Инициализация датасета
        
        Args:
            group_size: Размер группы (количество членов группы)
        """
        self.group_size = group_size
        self.members = []
        self.friends_network = nx.Graph()
        self.ip_mapping = {}
        
    def generate_realistic_ips(self, count: int) -> List[str]:
        """Генерация реалистичных IP-адресов"""
        ips = []
        
        # Популярные диапазоны IP для российских провайдеров
        russian_ranges = [
            "95.31.0.0/16",    # Ростелеком
            "178.176.0.0/12",  # МТС
            "85.143.0.0/16",   # Билайн  
            "37.29.0.0/16",    # Мегафон
            "46.39.0.0/16",    # TTK
            "217.69.0.0/16",   # Corbina
            "195.98.0.0/16",   # MGTS
            "109.195.0.0/16",  # Дом.ру
        ]
        
        for _ in range(count):
            # Выбираем случайный диапазон
            range_cidr = random.choice(russian_ranges)
            network = ipaddress.IPv4Network(range_cidr, strict=False)
            
            # Генерируем случайный IP в этом диапазоне
            random_int = random.randint(0, network.num_addresses - 1)
            ip = str(network.network_address + random_int)
            ips.append(ip)
            
        return ips
    
    def create_members_dataset(self):
        """Создание датасета членов группы"""
        print("Создание датасета членов группы...")
        
        # Генерируем имена членов группы
        first_names = ["Александр", "Дмитрий", "Максим", "Сергей", "Андрей", 
                      "Алексей", "Артем", "Илья", "Кирилл", "Михаил",
                      "Анна", "Мария", "Елена", "Ольга", "Татьяна",
                      "Наталья", "Ирина", "Светлана", "Юлия", "Екатерина"]
        
        last_names = ["Иванов", "Петров", "Сидоров", "Козлов", "Волков",
                     "Смирнов", "Кузнецов", "Попов", "Васильев", "Соколов",
                     "Иванова", "Петрова", "Сидорова", "Козлова", "Волкова",
                     "Смирнова", "Кузнецова", "Попова", "Васильева", "Соколова"]
        
        # Генерируем IP-адреса для членов группы
        member_ips = self.generate_realistic_ips(self.group_size)
        
        for i in range(self.group_size):
            member_id = f"member_{i+1}"
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            ip = member_ips[i]
            
            member_data = {
                'id': member_id,
                'name': name,
                'ip': ip,
                'vk_id': random.randint(100000000, 999999999),
                'friends': [],
                'friends_of_friends': []
            }
            
            self.members.append(member_data)
            self.ip_mapping[member_id] = ip
            
        print(f"Создано {len(self.members)} членов группы")
    
    def generate_friends_network(self):
        """Генерация сети друзей"""
        print("Генерация сети друзей...")
        
        # Добавляем узлы для членов группы
        for member in self.members:
            self.friends_network.add_node(member['id'], 
                                        name=member['name'], 
                                        ip=member['ip'],
                                        node_type='group_member')
        
        # Создаем связи между членами группы (дружба внутри группы)
        member_ids = [m['id'] for m in self.members]
        
        # Каждый член группы дружит с 3-7 другими членами группы
        for member_id in member_ids:
            num_friends_in_group = random.randint(3, min(7, len(member_ids)-1))
            potential_friends = [mid for mid in member_ids if mid != member_id]
            friends_in_group = random.sample(potential_friends, num_friends_in_group)
            
            for friend_id in friends_in_group:
                if not self.friends_network.has_edge(member_id, friend_id):
                    self.friends_network.add_edge(member_id, friend_id)
        
        # Генерируем внешних друзей для каждого члена группы
        external_friend_counter = 1
        
        for member in self.members:
            member_id = member['id']
            
            # Каждый член группы имеет 5-15 внешних друзей
            num_external_friends = random.randint(5, 15)
            external_friend_ips = self.generate_realistic_ips(num_external_friends)
            
            for i in range(num_external_friends):
                friend_id = f"external_friend_{external_friend_counter}"
                friend_ip = external_friend_ips[i]
                friend_name = f"Друг_{external_friend_counter}"
                friend_vk_id = random.randint(100000000, 999999999)
                
                # Добавляем внешнего друга в сеть
                self.friends_network.add_node(friend_id,
                                            name=friend_name,
                                            ip=friend_ip,
                                            vk_id=friend_vk_id,
                                            node_type='external_friend')
                
                # Создаем связь с членом группы
                self.friends_network.add_edge(member_id, friend_id)
                
                # Добавляем в список друзей члена группы
                member['friends'].append({
                    'id': friend_id,
                    'name': friend_name,
                    'ip': friend_ip,
                    'vk_id': random.randint(100000000, 999999999)
                })
                
                self.ip_mapping[friend_id] = friend_ip
                external_friend_counter += 1
        
        # Создаем связи между внешними друзьями (друзья друзей)
        external_friends = [n for n in self.friends_network.nodes() 
                          if self.friends_network.nodes[n]['node_type'] == 'external_friend']
        
        # Случайные связи между внешними друзьями
        num_external_connections = len(external_friends) // 3
        for _ in range(num_external_connections):
            friend1, friend2 = random.sample(external_friends, 2)
            if not self.friends_network.has_edge(friend1, friend2):
                self.friends_network.add_edge(friend1, friend2)
        
        print(f"Создана сеть из {self.friends_network.number_of_nodes()} узлов и {self.friends_network.number_of_edges()} связей")
    
    def create_induced_subgraphs(self) -> Dict[str, nx.Graph]:
        """Создание порожденных подграфов для различных подмножеств узлов"""
        print("Создание порожденных подграфов...")
        
        subgraphs = {}
        
        # 1. Порожденный подграф только членов группы
        group_member_ids = [m['id'] for m in self.members]
        group_subgraph = self.friends_network.subgraph(group_member_ids).copy()
        subgraphs['group_only'] = group_subgraph
        print(f"  - Подграф членов группы: {group_subgraph.number_of_nodes()} узлов, {group_subgraph.number_of_edges()} связей")
        
        # 2. Порожденные подграфы для каждого члена группы и его друзей
        for member in self.members:
            member_id = member['id']
            # Получаем всех соседей (друзей) данного члена группы
            neighbors = list(self.friends_network.neighbors(member_id))
            # Создаем подграф из члена группы и всех его друзей
            subgraph_nodes = [member_id] + neighbors
            member_subgraph = self.friends_network.subgraph(subgraph_nodes).copy()
            subgraphs[f'member_{member_id}'] = member_subgraph
        
        # 3. Порожденный подграф k-ядра (k-core)
        try:
            # Находим наибольшее k для k-ядра
            k_core = nx.k_core(self.friends_network)
            if len(k_core.nodes()) > 0:
                subgraphs['k_core'] = k_core
                print(f"  - K-ядро: {k_core.number_of_nodes()} узлов, {k_core.number_of_edges()} связей")
        except:
            print("  - K-ядро не найдено")
        
        # 4. Порожденный подграф высокой центральности
        # Берем топ-30% узлов по центральности степени
        degree_centrality = nx.degree_centrality(self.friends_network)
        top_nodes_count = max(10, len(self.friends_network.nodes()) // 3)
        top_central_nodes = sorted(degree_centrality.keys(), 
                                 key=lambda x: degree_centrality[x], 
                                 reverse=True)[:top_nodes_count]
        high_centrality_subgraph = self.friends_network.subgraph(top_central_nodes).copy()
        subgraphs['high_centrality'] = high_centrality_subgraph
        print(f"  - Подграф высокой центральности: {high_centrality_subgraph.number_of_nodes()} узлов, {high_centrality_subgraph.number_of_edges()} связей")
        
        return subgraphs
    
    def analyze_subgraph_properties(self, subgraphs: Dict[str, nx.Graph]) -> Dict:
        """Анализ свойств порожденных подграфов"""
        print("Анализ свойств порожденных подграфов...")
        
        analysis = {}
        
        for name, subgraph in subgraphs.items():
            if subgraph.number_of_nodes() == 0:
                continue
                
            properties = {
                'nodes': subgraph.number_of_nodes(),
                'edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph),
                'is_connected': nx.is_connected(subgraph),
            }
            
            # Добавляем метрики только для связных графов
            if nx.is_connected(subgraph):
                properties['diameter'] = nx.diameter(subgraph)
                properties['average_path_length'] = nx.average_shortest_path_length(subgraph)
            else:
                # Для несвязных графов анализируем наибольшую компоненту
                largest_cc = max(nx.connected_components(subgraph), key=len)
                largest_subgraph = subgraph.subgraph(largest_cc)
                properties['largest_component_size'] = len(largest_cc)
                properties['num_components'] = nx.number_connected_components(subgraph)
                if len(largest_cc) > 1:
                    properties['diameter'] = nx.diameter(largest_subgraph)
                    properties['average_path_length'] = nx.average_shortest_path_length(largest_subgraph)
            
            # Кластеризация
            if subgraph.number_of_nodes() > 2:
                properties['average_clustering'] = nx.average_clustering(subgraph)
            
            analysis[name] = properties
        
        return analysis
    
    def calculate_centrality_measures(self, subgraphs: Dict[str, nx.Graph] = None) -> Dict:
        """Вычисление мер центральности для основного графа и порожденных подграфов"""
        print("Вычисление мер центральности...")
        
        centrality_measures = {}
        
        # Анализируем основной граф
        print("  - Анализ основного графа...")
        main_centrality = self._calculate_single_graph_centrality(self.friends_network, "основной граф")
        centrality_measures['main_graph'] = main_centrality
        
        # Анализируем порожденные подграфы
        if subgraphs:
            print("  - Анализ порожденных подграфов...")
            for name, subgraph in subgraphs.items():
                if subgraph.number_of_nodes() > 1:
                    subgraph_centrality = self._calculate_single_graph_centrality(subgraph, name)
                    centrality_measures[f'subgraph_{name}'] = subgraph_centrality
        
        return centrality_measures
    
    def _calculate_single_graph_centrality(self, graph: nx.Graph, graph_name: str) -> Dict:
        """Вычисление мер центральности для одного графа"""
        centrality = {}
        
        # Центральность по посредничеству (Betweenness Centrality)
        betweenness = nx.betweenness_centrality(graph)
        centrality['betweenness'] = betweenness
        
        # Центральность по близости (Closeness Centrality)
        closeness = nx.closeness_centrality(graph)
        centrality['closeness'] = closeness
        
        # Центральность собственного вектора (Eigenvector Centrality)
        try:
            eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
            centrality['eigenvector'] = eigenvector
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            # Используем PageRank как альтернативу
            pagerank = nx.pagerank(graph)
            centrality['eigenvector'] = pagerank
        
        # Центральность по степени (Degree Centrality)
        degree = nx.degree_centrality(graph)
        centrality['degree'] = degree
        
        return centrality
    
    def analyze_group_members_centrality(self, centrality_measures: Dict, subgraph_analysis: Dict = None) -> pd.DataFrame:
        """Анализ центральности только для членов группы"""
        print("Анализ центральности членов группы...")
        
        group_member_ids = [m['id'] for m in self.members]
        
        analysis_data = []
        for member_id in group_member_ids:
            member_data = next(m for m in self.members if m['id'] == member_id)
            
            # Основные метрики из главного графа
            main_centrality = centrality_measures['main_graph']
            row = {
                'member_id': member_id,
                'name': member_data['name'],
                'ip': member_data['ip'],
                'vk_id': member_data['vk_id'],
                'betweenness_centrality': main_centrality['betweenness'][member_id],
                'closeness_centrality': main_centrality['closeness'][member_id],
                'eigenvector_centrality': main_centrality['eigenvector'][member_id],
                'degree_centrality': main_centrality['degree'][member_id],
                'num_friends': len(member_data['friends'])
            }
            
            # Добавляем метрики из подграфа только членов группы
            if 'subgraph_group_only' in centrality_measures and member_id in centrality_measures['subgraph_group_only']['betweenness']:
                group_centrality = centrality_measures['subgraph_group_only']
                row['group_betweenness'] = group_centrality['betweenness'][member_id]
                row['group_closeness'] = group_centrality['closeness'][member_id]
                row['group_eigenvector'] = group_centrality['eigenvector'][member_id]
                row['group_degree'] = group_centrality['degree'][member_id]
            
            analysis_data.append(row)
        
        df = pd.DataFrame(analysis_data)
        
        # Сортируем по различным мерам центральности
        print("\n=== АНАЛИЗ ЦЕНТРАЛЬНОСТИ В ОСНОВНОМ ГРАФЕ ===")
        print("\nТоп-5 по центральности посредничества:")
        top_betweenness = df.nlargest(5, 'betweenness_centrality')[['name', 'ip', 'vk_id', 'betweenness_centrality']]
        print(top_betweenness.to_string(index=False))
        
        print("\nТоп-5 по центральности близости:")
        top_closeness = df.nlargest(5, 'closeness_centrality')[['name', 'ip', 'vk_id', 'closeness_centrality']]
        print(top_closeness.to_string(index=False))
        
        print("\nТоп-5 по центральности собственного вектора:")
        top_eigenvector = df.nlargest(5, 'eigenvector_centrality')[['name', 'ip', 'vk_id', 'eigenvector_centrality']]
        print(top_eigenvector.to_string(index=False))
        
        # Анализ в подграфе группы (если доступен)
        if 'group_betweenness' in df.columns:
            print("\n=== АНАЛИЗ ЦЕНТРАЛЬНОСТИ В ПОДГРАФЕ ГРУППЫ ===")
            print("\nТоп-5 по центральности посредничества в группе:")
            top_group_betweenness = df.nlargest(5, 'group_betweenness')[['name', 'ip', 'vk_id', 'group_betweenness']]
            print(top_group_betweenness.to_string(index=False))
            
            print("\nТоп-5 по центральности близости в группе:")
            top_group_closeness = df.nlargest(5, 'group_closeness')[['name', 'ip', 'vk_id', 'group_closeness']]
            print(top_group_closeness.to_string(index=False))
        
        return df
    
    def visualize_network(self, centrality_measures: Dict, save_path: str = "vk_friends_network.png"):
        """Визуализация сети с выделением центральных узлов"""
        print("Создание визуализации сети...")
        
        plt.figure(figsize=(15, 12))
        
        # Позиционирование узлов
        pos = nx.spring_layout(self.friends_network, k=1, iterations=50)
        
        # Разделяем узлы на группы
        group_members = [n for n in self.friends_network.nodes() 
                        if self.friends_network.nodes[n]['node_type'] == 'group_member']
        external_friends = [n for n in self.friends_network.nodes() 
                          if self.friends_network.nodes[n]['node_type'] == 'external_friend']
        
        # Используем центральность из основного графа
        main_centrality = centrality_measures['main_graph']
        
        # Размеры узлов на основе центральности по посредничеству
        node_sizes = []
        for node in self.friends_network.nodes():
            size = 300 + main_centrality['betweenness'][node] * 2000
            node_sizes.append(size)
        
        # Цвета узлов на основе центральности собственного вектора
        node_colors = []
        for node in self.friends_network.nodes():
            color = main_centrality['eigenvector'][node]
            node_colors.append(color)
        
        # Рисуем рёбра
        nx.draw_networkx_edges(self.friends_network, pos, alpha=0.3, width=0.5, edge_color='gray')
        
        # Рисуем узлы членов группы
        nx.draw_networkx_nodes(self.friends_network, pos, 
                             nodelist=group_members,
                             node_size=[node_sizes[i] for i, n in enumerate(self.friends_network.nodes()) if n in group_members],
                             node_color=[node_colors[i] for i, n in enumerate(self.friends_network.nodes()) if n in group_members],
                             cmap=plt.cm.Reds, alpha=0.8, edgecolors='black', linewidths=2)
        
        # Рисуем узлы внешних друзей
        nx.draw_networkx_nodes(self.friends_network, pos,
                             nodelist=external_friends,
                             node_size=[node_sizes[i] for i, n in enumerate(self.friends_network.nodes()) if n in external_friends],
                             node_color=[node_colors[i] for i, n in enumerate(self.friends_network.nodes()) if n in external_friends],
                             cmap=plt.cm.Blues, alpha=0.6)
        
        # Добавляем подписи для членов группы
        group_labels = {}
        for node in group_members:
            member_data = next(m for m in self.members if m['id'] == node)
            group_labels[node] = member_data['name'].split()[0]  # Только имя
        
        nx.draw_networkx_labels(self.friends_network, pos, group_labels, font_size=8)
        
        plt.title("Сеть друзей ВКонтакте\n" + 
                 "Красные узлы - члены группы, синие - внешние друзья\n" +
                 "Размер узла = центральность посредничества, цвет = центральность собственного вектора",
                 fontsize=14, pad=20)
        
        # Добавляем цветовую шкалу
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                  norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Центральность собственного вектора', rotation=270, labelpad=15)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Визуализация сохранена в {save_path}")
        
        return save_path
    
    def save_dataset(self, filename: str = "vk_friends_dataset.json"):
        """Сохранение датасета в JSON файл"""
        dataset = {
            'group_members': self.members,
            'network_stats': {
                'total_nodes': self.friends_network.number_of_nodes(),
                'total_edges': self.friends_network.number_of_edges(),
                'group_size': self.group_size
            },
            'ip_mapping': self.ip_mapping
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Датасет сохранен в {filename}")
        return filename

def main():
    """Основная функция для выполнения анализа"""
    print("=== АНАЛИЗ СЕТИ ДРУЗЕЙ ВКОНТАКТЕ С МЕТОДОМ ПОРОЖДЕННОГО ГРАФА ===\n")
    
    # Создаем датасет
    dataset = VKFriendsIPDataset(group_size=20)
    
    # Генерируем данные
    dataset.create_members_dataset()
    dataset.generate_friends_network()
    
    # Создаем порожденные подграфы
    subgraphs = dataset.create_induced_subgraphs()
    
    # Анализируем свойства подграфов
    subgraph_analysis = dataset.analyze_subgraph_properties(subgraphs)
    
    print("\n=== СВОЙСТВА ПОРОЖДЕННЫХ ПОДГРАФОВ ===")
    for name, props in subgraph_analysis.items():
        print(f"\n{name.upper()}:")
        print(f"  Узлы: {props['nodes']}, Рёбра: {props['edges']}")
        print(f"  Плотность: {props['density']:.4f}")
        print(f"  Связность: {'Да' if props['is_connected'] else 'Нет'}")
        if 'diameter' in props:
            print(f"  Диаметр: {props['diameter']}")
            print(f"  Средняя длина пути: {props['average_path_length']:.4f}")
        if 'average_clustering' in props:
            print(f"  Средний коэффициент кластеризации: {props['average_clustering']:.4f}")
        if 'num_components' in props:
            print(f"  Количество компонент связности: {props['num_components']}")
    
    # Вычисляем меры центральности для основного графа и подграфов
    centrality_measures = dataset.calculate_centrality_measures(subgraphs)
    
    # Анализируем центральность членов группы
    analysis_df = dataset.analyze_group_members_centrality(centrality_measures, subgraph_analysis)
    
    # Сохраняем результаты анализа
    analysis_df.to_csv('group_members_centrality_analysis.csv', index=False, encoding='utf-8')
    print(f"\nАнализ центральности сохранен в group_members_centrality_analysis.csv")
    
    # Сохраняем анализ подграфов
    subgraph_df = pd.DataFrame(subgraph_analysis).T
    subgraph_df.to_csv('subgraph_analysis.csv', encoding='utf-8')
    print(f"Анализ подграфов сохранен в subgraph_analysis.csv")
    
    # Создаем визуализацию
    viz_path = dataset.visualize_network(centrality_measures)
    
    # Сохраняем датасет
    dataset_path = dataset.save_dataset()
    
    print(f"\n=== ОБЩИЕ РЕЗУЛЬТАТЫ ===")
    print(f"Общее количество узлов в сети: {dataset.friends_network.number_of_nodes()}")
    print(f"Общее количество связей: {dataset.friends_network.number_of_edges()}")
    print(f"Членов группы: {dataset.group_size}")
    print(f"Внешних друзей: {dataset.friends_network.number_of_nodes() - dataset.group_size}")
    print(f"Плотность основной сети: {nx.density(dataset.friends_network):.4f}")
    
    # Выводим самых центральных членов группы
    print(f"\n=== САМЫЕ ЦЕНТРАЛЬНЫЕ ЧЛЕНЫ ГРУППЫ В ОСНОВНОЙ СЕТИ ===")
    
    most_central_betweenness = analysis_df.loc[analysis_df['betweenness_centrality'].idxmax()]
    print(f"Наивысшая центральность посредничества: {most_central_betweenness['name']} (VK ID: {most_central_betweenness['vk_id']}, IP: {most_central_betweenness['ip']}) - {most_central_betweenness['betweenness_centrality']:.4f}")
    
    most_central_closeness = analysis_df.loc[analysis_df['closeness_centrality'].idxmax()]
    print(f"Наивысшая центральность близости: {most_central_closeness['name']} (VK ID: {most_central_closeness['vk_id']}, IP: {most_central_closeness['ip']}) - {most_central_closeness['closeness_centrality']:.4f}")
    
    most_central_eigenvector = analysis_df.loc[analysis_df['eigenvector_centrality'].idxmax()]
    print(f"Наивысшая центральность собственного вектора: {most_central_eigenvector['name']} (VK ID: {most_central_eigenvector['vk_id']}, IP: {most_central_eigenvector['ip']}) - {most_central_eigenvector['eigenvector_centrality']:.4f}")
    
    # Анализ в подграфе группы
    if 'group_betweenness' in analysis_df.columns:
        print(f"\n=== САМЫЕ ЦЕНТРАЛЬНЫЕ ЧЛЕНЫ ВНУТРИ ГРУППЫ ===")
        
        most_central_group_betweenness = analysis_df.loc[analysis_df['group_betweenness'].idxmax()]
        print(f"Наивысшая центральность посредничества в группе: {most_central_group_betweenness['name']} (VK ID: {most_central_group_betweenness['vk_id']}) - {most_central_group_betweenness['group_betweenness']:.4f}")
        
        most_central_group_closeness = analysis_df.loc[analysis_df['group_closeness'].idxmax()]
        print(f"Наивысшая центральность близости в группе: {most_central_group_closeness['name']} (VK ID: {most_central_group_closeness['vk_id']}) - {most_central_group_closeness['group_closeness']:.4f}")

if __name__ == "__main__":
    main()