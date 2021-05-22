# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class NetworkxUtil:
    def plot_structure_model(self, structure_model, layout_method="spring", layout_seed=1, figsize=(5,5), 
                              node_shape="o", node_size=1500, node_color="#1abc9c", font_size=18, 
                             edge_color="#34495e", min_edge_width=2, max_edge_width =3.5, arrowsize=25, 
                              plot_edge_weights=True, edge_weights_color="#e74c3c", edge_weights_fontsize=12, alpha=0.8):
      # plot settings
      fig, ax = plt.subplots(figsize=figsize)
      pos = self._load_layout(structure_model=structure_model, layout_method=layout_method, layout_seed=layout_seed)

      # edgeの重み係数をplotするか否かのフラグ
      if plot_edge_weights:
        nx.draw_networkx_edge_labels(
            structure_model, pos,
            edge_labels={(u, v): round(d["weight"], 4) for (u,v,d) in structure_model.edges(data=True)},
            font_color=edge_weights_color,
            font_size = edge_weights_fontsize
        )

      # edgeの重みに応じて太さを変更、最低でもmin_edge_widthに設定
      edge_width = [
                    np.min([np.max([d["weight"], min_edge_width]), max_edge_width] )
                    for (u, v, d) in structure_model.edges(data=True)
      ]
      # networkxでの推定したモデルのplot
      nx.draw_networkx(
          structure_model,
          ax = ax,
          pos = pos,
          node_shape = node_shape,
          node_color = node_color,
          node_size = node_size,
          edge_color = edge_color,
          width = edge_width,
          alpha = alpha, 
          font_size = font_size,
          arrowsize = arrowsize,
          with_labels = True,
          # connectionstyle="arc3, rad=0.1" # if curve
      )
      plt.axis("off")
      fig.show()

    def _load_layout(self, structure_model, layout_method, layout_seed=0):
      """layout_methodを指定して、pos: position keyed by nodeを取得"""
      if layout_method == "circular":
        pos = nx.circular_layout(structure_model)
      elif layout_method == "spring":
        pos = nx.spring_layout(structure_model, seed=layout_seed)
      elif layout_method == "planar":
        pos = nx.planar_layout(structure_model)
      elif layout_method == "shell":
        pos = nx.shell_layout(structure_model)
      elif layout_method == "random":
        pos = nx.random_layout(structure_model, seed=layout_seed)
      else:
        raise ValueError(f"Method {layout_method} is not expected.")
      return pos

    @staticmethod
    def print_weights(structure_model, cutoff=0):
      """エッジとその重みをprint"""
      for (u, v, d) in structure_model.edges(data=True):
        if np.abs(d["weight"]) >= cutoff:
          print(f"[ {u} ] ---> [ {v} ]\t\tWeight\t{d['weight']:.5f}")