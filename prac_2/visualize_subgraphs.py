#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дополнительная визуализация порожденных подграфов
"""

import networkx as nx
import matplotlib.pyplot as plt
import json
from vk_friends_ip_dataset import VKFriendsIPDataset

def visualize_subgraphs_comparison():
    """Создание сравнительной визуализации порожденных подграфов"""
    
    # Создаем датасет
    dataset = VKFriendsIPDataset(group_size=20)
    dataset.create_members_dataset()
    dataset.generate_friends_network()
    
    # Создаем порожденные подграфы
    subgraphs = dataset.create_induced_subgraphs()
    
    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Сравнение порожденных подграфов сети ВКонтакте', fontsize=16, fontweight='bold')
    
    # 1. Подграф только членов группы
    ax1 = axes[0, 0]
    group_subgraph = subgraphs['group_only']
    pos1 = nx.spring_layout(group_subgraph, k=1, iterations=50)
    
    # Размеры узлов пропорциональны степени
    node_sizes1 = [300 + group_subgraph.degree(node) * 50 for node in group_subgraph.nodes()]
    
    nx.draw(group_subgraph, pos1, ax=ax1, 
            node_size=node_sizes1, 
            node_color='lightcoral', 
            edge_color='gray', 
            alpha=0.8,
            with_labels=False)
    
    ax1.set_title(f'Подграф членов группы\n{group_subgraph.number_of_nodes()} узлов, {group_subgraph.number_of_edges()} связей\nПлотность: {nx.density(group_subgraph):.3f}')
    
    # 2. K-ядро
    ax2 = axes[0, 1]
    if 'k_core' in subgraphs:
        k_core_subgraph = subgraphs['k_core']
        pos2 = nx.spring_layout(k_core_subgraph, k=1, iterations=50)
        
        node_sizes2 = [300 + k_core_subgraph.degree(node) * 50 for node in k_core_subgraph.nodes()]
        
        nx.draw(k_core_subgraph, pos2, ax=ax2,
                node_size=node_sizes2,
                node_color='lightblue',
                edge_color='gray',
                alpha=0.8,
                with_labels=False)
        
        ax2.set_title(f'K-ядро\n{k_core_subgraph.number_of_nodes()} узлов, {k_core_subgraph.number_of_edges()} связей\nПлотность: {nx.density(k_core_subgraph):.3f}')
    else:
        ax2.text(0.5, 0.5, 'K-ядро не найдено', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('K-ядро')
    
    # 3. Подграф высокой центральности
    ax3 = axes[1, 0]
    high_cent_subgraph = subgraphs['high_centrality']
    pos3 = nx.spring_layout(high_cent_subgraph, k=0.5, iterations=50)
    
    # Разделяем узлы на членов группы и внешних друзей
    group_members = [m['id'] for m in dataset.members]
    group_nodes_in_subgraph = [n for n in high_cent_subgraph.nodes() if n in group_members]
    external_nodes_in_subgraph = [n for n in high_cent_subgraph.nodes() if n not in group_members]
    
    # Рисуем членов группы красным, внешних друзей синим
    if group_nodes_in_subgraph:
        nx.draw_networkx_nodes(high_cent_subgraph, pos3, 
                             nodelist=group_nodes_in_subgraph,
                             node_color='red', 
                             node_size=200,
                             alpha=0.8,
                             ax=ax3)
    
    if external_nodes_in_subgraph:
        nx.draw_networkx_nodes(high_cent_subgraph, pos3,
                             nodelist=external_nodes_in_subgraph,
                             node_color='lightblue',
                             node_size=100,
                             alpha=0.6,
                             ax=ax3)
    
    nx.draw_networkx_edges(high_cent_subgraph, pos3, ax=ax3, alpha=0.3, edge_color='gray')
    
    ax3.set_title(f'Подграф высокой центральности\n{high_cent_subgraph.number_of_nodes()} узлов, {high_cent_subgraph.number_of_edges()} связей\nПлотность: {nx.density(high_cent_subgraph):.3f}')
    
    # 4. Пример подграфа одного члена группы и его друзей
    ax4 = axes[1, 1]
    member_key = f'member_{dataset.members[0]["id"]}'
    if member_key in subgraphs:
        member_subgraph = subgraphs[member_key]
        pos4 = nx.spring_layout(member_subgraph, k=1, iterations=50)
        
        # Центральный узел (член группы) выделяем
        central_node = dataset.members[0]['id']
        other_nodes = [n for n in member_subgraph.nodes() if n != central_node]
        
        # Рисуем центральный узел
        nx.draw_networkx_nodes(member_subgraph, pos4,
                             nodelist=[central_node],
                             node_color='red',
                             node_size=500,
                             ax=ax4)
        
        # Рисуем остальные узлы
        nx.draw_networkx_nodes(member_subgraph, pos4,
                             nodelist=other_nodes,
                             node_color='lightgreen',
                             node_size=200,
                             alpha=0.7,
                             ax=ax4)
        
        nx.draw_networkx_edges(member_subgraph, pos4, ax=ax4, alpha=0.5, edge_color='gray')
        
        ax4.set_title(f'Подграф {dataset.members[0]["name"]}\nи его друзей\n{member_subgraph.number_of_nodes()} узлов, {member_subgraph.number_of_edges()} связей')
    
    # Убираем оси у всех подграфиков
    for ax in axes.flat:
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig('subgraphs_comparison.png', dpi=300, bbox_inches='tight')
    print("Сравнительная визуализация подграфов сохранена в subgraphs_comparison.png")
    
    return 'subgraphs_comparison.png'

if __name__ == "__main__":
    visualize_subgraphs_comparison()