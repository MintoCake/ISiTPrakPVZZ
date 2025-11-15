#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Датасет друзей ВКонтакте и анализ центральности сети
Задача №2: Анализ информации о друзьях и друзьях друзей из ВК для членов группы
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import json
import random

class VKFriendsDataset:
    """Класс для анализа датасета друзей ВКонтакте из databaseFriends.json"""
    
    def __init__(self, json_file: str = "databaseFriends.json"):
        """
        Инициализация датасета
        
        Args:
            json_file: Путь к JSON файлу с данными
        """
        self.json_file = json_file
        self.members = []
        self.friends_network = nx.Graph()
        self.all_people = {}  # Словарь всех людей по id для быстрого доступа
        
    def load_dataset(self):
        """Загрузка датасета из JSON файла"""
        print(f"Загрузка датасета из {self.json_file}...")
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Основные члены группы - это корневые объекты в массиве
        self.members = data
        
        # Строим словарь всех людей для быстрого доступа
        self._build_people_dict(data)
        
        print(f"Загружено {len(self.members)} членов группы")
    
    def _build_people_dict(self, data: List[Dict], visited: set = None):
        """Рекурсивное построение словаря всех людей"""
        if visited is None:
            visited = set()
        
        for person in data:
            person_id = str(person['id'])  # Преобразуем id в строку для единообразия
            
            if person_id not in visited:
                visited.add(person_id)
                self.all_people[person_id] = {
                    'id': person_id,
                    'first_name': person.get('first_name', ''),
                    'last_name': person.get('last_name', ''),
                    'city': person.get('city', ''),
                    'is_group_member': person_id in [str(m['id']) for m in self.members]
                }
                
                # Рекурсивно обрабатываем друзей
                if 'friends' in person and person['friends']:
                    self._build_people_dict(person['friends'], visited)
    
    def build_friends_network(self):
        """Построение сети друзей на основе данных из JSON"""
        print("Построение сети друзей...")
        
        def add_person_to_network(person: Dict, is_group_member: bool = False):
            """Рекурсивное добавление человека и его друзей в сеть"""
            person_id = str(person['id'])
            
            # Добавляем узел, если его еще нет
            if person_id not in self.friends_network:
                self.friends_network.add_node(
                    person_id,
                    first_name=person.get('first_name', ''),
                    last_name=person.get('last_name', ''),
                    city=person.get('city', ''),
                    node_type='group_member' if is_group_member else 'friend'
                )
            
            # Добавляем связи с друзьями
            if 'friends' in person and person['friends']:
                for friend in person['friends']:
                    friend_id = str(friend['id'])
                    
                    # Добавляем друга как узел
                    if friend_id not in self.friends_network:
                        self.friends_network.add_node(
                            friend_id,
                            first_name=friend.get('first_name', ''),
                            last_name=friend.get('last_name', ''),
                            city=friend.get('city', ''),
                            node_type='friend'
                        )
                    
                    # Добавляем связь (если еще нет)
                    if not self.friends_network.has_edge(person_id, friend_id):
                        self.friends_network.add_edge(person_id, friend_id)
                    
                    # Рекурсивно обрабатываем друзей друзей
                    if 'friends' in friend and friend['friends']:
                        add_person_to_network(friend, is_group_member=False)
        
        # Обрабатываем всех членов группы
        for member in self.members:
            add_person_to_network(member, is_group_member=True)
        
        # Добавляем связи между членами группы (5-10 общих связей)
        self._add_group_member_connections()
        
        # Добавляем сложные связи через промежуточных друзей
        self._add_complex_connections()
        
        print(f"Создана сеть из {self.friends_network.number_of_nodes()} узлов и {self.friends_network.number_of_edges()} связей")
    
    def _add_group_member_connections(self):
        """Добавление связей между членами группы (5-10 общих связей)"""
        group_member_ids = [str(m['id']) for m in self.members]
        
        if len(group_member_ids) < 2:
            return
        
        # Определяем количество связей (5-10)
        num_connections = random.randint(5, 10)
        
        # Создаем список всех возможных пар
        possible_pairs = []
        for i in range(len(group_member_ids)):
            for j in range(i + 1, len(group_member_ids)):
                possible_pairs.append((group_member_ids[i], group_member_ids[j]))
        
        # Случайно выбираем пары для создания связей
        # Убеждаемся, что не создаем дубликаты
        connections_added = 0
        used_pairs = set()
        
        while connections_added < num_connections and len(used_pairs) < len(possible_pairs):
            pair = random.choice(possible_pairs)
            
            if pair not in used_pairs:
                member1_id, member2_id = pair
                
                # Добавляем связь, если её еще нет
                if not self.friends_network.has_edge(member1_id, member2_id):
                    self.friends_network.add_edge(member1_id, member2_id)
                    connections_added += 1
                    used_pairs.add(pair)
        
        print(f"Добавлено {connections_added} связей между членами группы")
    
    def _add_complex_connections(self):
        """Добавление сложных связей через промежуточных друзей для создания путей между основными пользователями"""
        group_member_ids = [str(m['id']) for m in self.members]
        
        if len(group_member_ids) < 2:
            return
        
        connections_added = 0
        
        # Получаем друзей каждого члена группы
        member_friends = {}
        for member_id in group_member_ids:
            friends = list(self.friends_network.neighbors(member_id))
            # Фильтруем только друзей (не членов группы)
            friends = [f for f in friends if f not in group_member_ids]
            member_friends[member_id] = friends
        
        # Создаем связи между друзьями разных членов группы
        # Это создаст пути: основной_пользователь1 -> друг1 -> друг2 -> основной_пользователь2
        for i, member1_id in enumerate(group_member_ids):
            for member2_id in group_member_ids[i+1:]:
                friends1 = member_friends.get(member1_id, [])
                friends2 = member_friends.get(member2_id, [])
                
                if not friends1 or not friends2:
                    continue
                
                # Создаем 1-3 связи между друзьями разных членов группы
                num_connections = random.randint(1, 3)
                for _ in range(num_connections):
                    if friends1 and friends2:
                        friend1 = random.choice(friends1)
                        friend2 = random.choice(friends2)
                        
                        # Добавляем связь между друзьями, если её еще нет
                        if not self.friends_network.has_edge(friend1, friend2):
                            self.friends_network.add_edge(friend1, friend2)
                            connections_added += 1
        
        # Создаем связи между друзьями друзей (друзьями второго уровня)
        # Это создаст более длинные пути: основной_пользователь1 -> друг1 -> друг_друга1 -> друг_друга2 -> друг2 -> основной_пользователь2
        friends_of_friends_connections = 0
        
        # Для каждого члена группы берем его друзей
        for member_id in group_member_ids:
            friends = member_friends.get(member_id, [])
            
            # Для каждого друга находим его друзей (друзья друзей)
            for friend_id in friends:
                if friend_id not in self.friends_network:
                    continue
                
                friends_of_friend = list(self.friends_network.neighbors(friend_id))
                # Исключаем самого члена группы и других членов группы
                friends_of_friend = [f for f in friends_of_friend 
                                   if f != member_id and f not in group_member_ids]
                
                # Создаем связи между друзьями друзей разных членов группы
                for other_member_id in group_member_ids:
                    if other_member_id == member_id:
                        continue
                    
                    other_friends = member_friends.get(other_member_id, [])
                    
                    # Создаем 1-2 связи между друзьями друзей
                    num_connections = random.randint(0, 2)
                    for _ in range(num_connections):
                        if friends_of_friend and other_friends:
                            friend_of_friend = random.choice(friends_of_friend)
                            other_friend = random.choice(other_friends)
                            
                            # Добавляем связь, если её еще нет
                            if not self.friends_network.has_edge(friend_of_friend, other_friend):
                                self.friends_network.add_edge(friend_of_friend, other_friend)
                                friends_of_friends_connections += 1
        
        total_connections = connections_added + friends_of_friends_connections
        print(f"Добавлено {total_connections} сложных связей через промежуточных друзей ({connections_added} между друзьями, {friends_of_friends_connections} между друзьями друзей)")
    
    def create_induced_subgraphs(self) -> Dict[str, nx.Graph]:
        """Создание порожденных подграфов для различных подмножеств узлов"""
        print("Создание порожденных подграфов...")
        
        subgraphs = {}
        
        # 1. Порожденный подграф только членов группы
        group_member_ids = [str(m['id']) for m in self.members]
        group_subgraph = self.friends_network.subgraph(group_member_ids).copy()
        subgraphs['group_only'] = group_subgraph
        print(f"  - Подграф членов группы: {group_subgraph.number_of_nodes()} узлов, {group_subgraph.number_of_edges()} связей")
        
        # 2. Порожденные подграфы для каждого члена группы и его друзей
        for member in self.members:
            member_id = str(member['id'])
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
        
        group_member_ids = [str(m['id']) for m in self.members]
        
        analysis_data = []
        for member in self.members:
            member_id = str(member['id'])
            
            # Основные метрики из главного графа
            main_centrality = centrality_measures['main_graph']
            
            # Получаем количество друзей из графа
            num_friends = len(list(self.friends_network.neighbors(member_id)))
            
            row = {
                'member_id': member_id,
                'first_name': member.get('first_name', ''),
                'last_name': member.get('last_name', ''),
                'name': f"{member.get('first_name', '')} {member.get('last_name', '')}".strip(),
                'city': member.get('city', ''),
                'betweenness_centrality': main_centrality['betweenness'].get(member_id, 0),
                'closeness_centrality': main_centrality['closeness'].get(member_id, 0),
                'eigenvector_centrality': main_centrality['eigenvector'].get(member_id, 0),
                'degree_centrality': main_centrality['degree'].get(member_id, 0),
                'num_friends': num_friends
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
        top_betweenness = df.nlargest(5, 'betweenness_centrality')[['name', 'city', 'betweenness_centrality']]
        print(top_betweenness.to_string(index=False))
        
        print("\nТоп-5 по центральности близости:")
        top_closeness = df.nlargest(5, 'closeness_centrality')[['name', 'city', 'closeness_centrality']]
        print(top_closeness.to_string(index=False))
        
        print("\nТоп-5 по центральности собственного вектора:")
        top_eigenvector = df.nlargest(5, 'eigenvector_centrality')[['name', 'city', 'eigenvector_centrality']]
        print(top_eigenvector.to_string(index=False))
        
        # Анализ в подграфе группы (если доступен)
        if 'group_betweenness' in df.columns:
            print("\n=== АНАЛИЗ ЦЕНТРАЛЬНОСТИ В ПОДГРАФЕ ГРУППЫ ===")
            print("\nТоп-5 по центральности посредничества в группе:")
            top_group_betweenness = df.nlargest(5, 'group_betweenness')[['name', 'city', 'group_betweenness']]
            print(top_group_betweenness.to_string(index=False))
            
            print("\nТоп-5 по центральности близости в группе:")
            top_group_closeness = df.nlargest(5, 'group_closeness')[['name', 'city', 'group_closeness']]
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
                          if self.friends_network.nodes[n]['node_type'] == 'friend']
        
        # Используем центральность из основного графа
        main_centrality = centrality_measures['main_graph']
        
        # Размеры узлов на основе центральности по посредничеству
        node_sizes = []
        for node in self.friends_network.nodes():
            size = 300 + main_centrality['betweenness'].get(node, 0) * 2000
            node_sizes.append(size)
        
        # Цвета узлов на основе центральности собственного вектора
        node_colors = []
        for node in self.friends_network.nodes():
            color = main_centrality['eigenvector'].get(node, 0)
            node_colors.append(color)
        
        # Рисуем рёбра
        nx.draw_networkx_edges(self.friends_network, pos, alpha=0.6, width=3.0, edge_color='gray')
        
        # Рисуем узлы членов группы
        if group_members:
            group_sizes = [node_sizes[i] for i, n in enumerate(self.friends_network.nodes()) if n in group_members]
            group_colors = [node_colors[i] for i, n in enumerate(self.friends_network.nodes()) if n in group_members]
        nx.draw_networkx_nodes(self.friends_network, pos, 
                             nodelist=group_members,
                                 node_size=group_sizes,
                                 node_color=group_colors,
                             cmap=plt.cm.Reds, alpha=0.8, edgecolors='black', linewidths=2)
        
        # Рисуем узлы внешних друзей
        if external_friends:
            friend_sizes = [node_sizes[i] for i, n in enumerate(self.friends_network.nodes()) if n in external_friends]
            # Используем фиксированный синий цвет для друзей, чтобы они были хорошо видны
        nx.draw_networkx_nodes(self.friends_network, pos,
                             nodelist=external_friends,
                                 node_size=friend_sizes,
                                 node_color='steelblue',
                                 alpha=0.8, edgecolors='darkblue', linewidths=1)
        
        # Добавляем подписи для членов группы
        group_labels = {}
        for node in group_members:
            node_data = self.friends_network.nodes[node]
            first_name = node_data.get('first_name', '')
            group_labels[node] = first_name
        
        nx.draw_networkx_labels(self.friends_network, pos, group_labels, font_size=8)
        
        plt.title("Сеть друзей ВКонтакте\n" + 
                 "Красные узлы - члены группы, синие - друзья\n" +
                 "Размер узла = центральность посредничества, цвет = центральность собственного вектора",
                 fontsize=14, pad=20)
        
        # Добавляем цветовую шкалу
        if node_colors:
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
                'group_size': len(self.members)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Датасет сохранен в {filename}")
        return filename

def main():
    """Основная функция для выполнения анализа"""
    print("=== АНАЛИЗ СЕТИ ДРУЗЕЙ ВКОНТАКТЕ С МЕТОДОМ ПОРОЖДЕННОГО ГРАФА ===\n")
    
    # Создаем датасет
    dataset = VKFriendsDataset(json_file="databaseFriends.json")
    
    # Загружаем данные
    dataset.load_dataset()
    
    # Строим сеть друзей
    dataset.build_friends_network()
    
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
    print(f"Членов группы: {len(dataset.members)}")
    print(f"Друзей: {dataset.friends_network.number_of_nodes() - len(dataset.members)}")
    print(f"Плотность основной сети: {nx.density(dataset.friends_network):.4f}")
    
    # Выводим самых центральных членов группы
    print(f"\n=== САМЫЕ ЦЕНТРАЛЬНЫЕ ЧЛЕНЫ ГРУППЫ В ОСНОВНОЙ СЕТИ ===")
    
    most_central_betweenness = analysis_df.loc[analysis_df['betweenness_centrality'].idxmax()]
    print(f"Наивысшая центральность посредничества: {most_central_betweenness['name']} (Город: {most_central_betweenness['city']}) - {most_central_betweenness['betweenness_centrality']:.4f}")
    
    most_central_closeness = analysis_df.loc[analysis_df['closeness_centrality'].idxmax()]
    print(f"Наивысшая центральность близости: {most_central_closeness['name']} (Город: {most_central_closeness['city']}) - {most_central_closeness['closeness_centrality']:.4f}")
    
    most_central_eigenvector = analysis_df.loc[analysis_df['eigenvector_centrality'].idxmax()]
    print(f"Наивысшая центральность собственного вектора: {most_central_eigenvector['name']} (Город: {most_central_eigenvector['city']}) - {most_central_eigenvector['eigenvector_centrality']:.4f}")
    
    # Анализ в подграфе группы
    if 'group_betweenness' in analysis_df.columns:
        print(f"\n=== САМЫЕ ЦЕНТРАЛЬНЫЕ ЧЛЕНЫ ВНУТРИ ГРУППЫ ===")
        
        most_central_group_betweenness = analysis_df.loc[analysis_df['group_betweenness'].idxmax()]
        print(f"Наивысшая центральность посредничества в группе: {most_central_group_betweenness['name']} (Город: {most_central_group_betweenness['city']}) - {most_central_group_betweenness['group_betweenness']:.4f}")
        
        most_central_group_closeness = analysis_df.loc[analysis_df['group_closeness'].idxmax()]
        print(f"Наивысшая центральность близости в группе: {most_central_group_closeness['name']} (Город: {most_central_group_closeness['city']}) - {most_central_group_closeness['group_closeness']:.4f}")

if __name__ == "__main__":
    main()
