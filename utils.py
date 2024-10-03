import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Polygon

def label_to_color(label):
    
    n_points = label.shape[0]
    color = np.zeros((n_points, 3))

    # color template (2021 pantone color: orbital)
    rgb = np.zeros((11, 3))
    rgb[0, :] = [253, 134, 18]
    rgb[1, :] = [106, 194, 217]
    rgb[2, :] = [111, 146, 110]
    rgb[3, :] = [153, 0, 17]
    rgb[4, :] = [179, 173, 151]
    rgb[5, :] = [245, 228, 0]
    rgb[6, :] = [255, 0, 0]
    rgb[7, :] = [0, 255, 0]
    rgb[8, :] = [0, 0, 255]
    rgb[9, :] = [18, 134, 253]
    rgb[10, :] = [155, 155, 155] # grey

    for idx_color in range(10):
        color[label == idx_color, :] = rgb[idx_color, :]
    return color

def PD_metric_to_ellipse(G, center, scale, **kwargs):
    
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # find angle of ellipse
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # draw ellipse
    width, height = 2 * scale * np.sqrt(eigvals)
    return Ellipse(xy=center, width=width, height=height, angle=np.degrees(theta), **kwargs)

def visualize_Riemannian_metric_as_ellipses(train_ds, pretrained_model, get_Riemannian_metric, device, at=None, figsize=(7, 7)):
    if at is not None:
        num_points_for_each_class = 200
        label_unique = torch.unique(train_ds.targets)
        
        z_ = []
        z_sampled_ = []
        label_ = []
        for label in label_unique:
            temp_data = train_ds.data[train_ds.targets == label][:num_points_for_each_class]
            temp_z = pretrained_model.encode(temp_data.to(device))
            z_.append(temp_z)
            label_.append(label.repeat(temp_z.size(0)))
        z_sampled_ = at
        G_ = get_Riemannian_metric(pretrained_model.decode, z_sampled_).detach().cpu()
        z_sampled_ = z_sampled_.detach().cpu().numpy()
        
        z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
        label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
        color_ = label_to_color(label_)
        
        plt.figure(figsize=figsize)
        plt.title('Latent space embeddings with equidistant ellipses')
        z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
        eig_mean = torch.svd(G_).S.mean().item()
        scale = 0.05 * z_scale * np.sqrt(eig_mean)
        alpha = 0.3

        for idx in range(len(z_sampled_)):
            e = PD_metric_to_ellipse(np.linalg.inv(G_[idx,:,:]), z_sampled_[idx,:], scale, fc="k", alpha=alpha)
            plt.gca().add_artist(e)
        for label in label_unique:
            label = label.item()
            plt.scatter(z_[label_==label,0], z_[label_==label,1], c=color_[label_==label]/255, label=label, s=5)
            
        plt.legend()
        plt.axis('equal')
        plt.show()
    else:
        num_points_for_each_class = 200
        num_G_plots_for_each_class = 20
        # num_G_plots = 100
        label_unique = torch.unique(train_ds.targets)
        
        z_ = []
        z_sampled_ = []
        label_ = []
        label_sampled_ = []
        G_ = []
        for label in label_unique:
            temp_data = train_ds.data[train_ds.targets == label][:num_points_for_each_class]
            temp_z = pretrained_model.encode(temp_data.to(device))
            z_sampled = temp_z[torch.randperm(len(temp_z))[:num_G_plots_for_each_class]]
            G = get_Riemannian_metric(pretrained_model.decode, z_sampled)

            z_.append(temp_z)
            label_.append(label.repeat(temp_z.size(0)))
            z_sampled_.append(z_sampled)
            label_sampled_.append(label.repeat(z_sampled.size(0)))
            G_.append(G)
        # z_temp = pretrained_model.encode(train_ds.data[:num_G_plots].to(device))
        # z_permuted = z_temp[torch.randperm(len(z_temp))]
        # eta = torch.rand(len(z_temp), 1).to(z_temp)
        # z_sampled_ = eta * z_temp + (1-eta) * z_permuted
        # G_ = get_Riemannian_metric(pretrained_model.decode, z_sampled_).detach().cpu()
        # z_sampled_ = z_sampled_.detach().cpu().numpy()
        
        z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
        label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
        color_ = label_to_color(label_)
        G_ = torch.cat(G_, dim=0).detach().cpu()
        z_sampled_ = torch.cat(z_sampled_, dim=0).detach().cpu().numpy()
        label_sampled_ = torch.cat(label_sampled_, dim=0).detach().cpu().numpy()
        color_sampled_ = label_to_color(label_sampled_)

        plt.figure(figsize=figsize)
        plt.title('Latent space embeddings with equidistant ellipses')
        z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
        eig_mean = torch.svd(G_).S.mean().item()
        scale = 0.05 * z_scale * np.sqrt(eig_mean)
        alpha = 0.3

        for idx in range(len(z_sampled_)):
            e = PD_metric_to_ellipse(np.linalg.inv(G_[idx,:,:]), z_sampled_[idx,:], scale, fc=color_sampled_[idx,:]/255.0, alpha=alpha)
            plt.gca().add_artist(e)
        for label in label_unique:
            label = label.item()
            plt.scatter(z_[label_==label,0], z_[label_==label,1], c=color_[label_==label]/255, label=label, s=5)
            
        plt.legend()
        plt.axis('equal')
        plt.show()

import math
import random
class relation_loader():
    def __init__(self, relations, negative_relations, neg_sample_num=50, batch_num=10):
        self.relations = relations
        self.negative_relations = negative_relations
        self.batch_num = batch_num
        self.neg_sample_num = neg_sample_num
        self.shuffle()
        self.iter = 0
        self.len = math.ceil(len(relations)/float(batch_num))
        
    def get_item(self, points):
        if self.iter == self.len - 1:
            relations = self.relations[self.batch_num * self.iter :]
            negative_relations = self.negative_relations[self.batch_num * self.iter :]
        else:
            relations = self.relations[self.batch_num * self.iter : self.batch_num * (self.iter + 1)]
            negative_relations = self.negative_relations[self.batch_num * self.iter : self.batch_num * (self.iter + 1)]
        pos_pairs = points[relations,:]
        negative_relations_sampled = [random.sample(negative_relation, self.neg_sample_num) for negative_relation in negative_relations]
        neg_pairs = points[negative_relations_sampled,:]
        self.iter += 1
        if self.iter == self.len:
            self.iter = 0
            self.shuffle()
        return pos_pairs, neg_pairs
        
    def shuffle(self):
        relation_zip = list(zip(self.relations, self.negative_relations))
        random.shuffle(relation_zip)
        self.relations, self.negative_relations = zip(*relation_zip)
        
def map_row(H1, H2, n):
    edge_mask = (H1 == 1.0)
    m         = np.sum(edge_mask).astype(int)
    assert m > 0
    d = H2
    sorted_dist = np.argsort(d)
    precs       = np.zeros(m)
    n_correct   = 0
    j = 0
    for i in range(1,n):
        if edge_mask[sorted_dist[i]]:
            n_correct += 1
            precs[j] = n_correct/float(i)
            j += 1
            if j == m:
                break
    return np.sum(precs)/m

def map_score(points, relation_graph, n, dist, mode):
    n_points = points.shape[0]
    p1 = points.repeat(n_points, 1, 1).reshape(-1,2)
    p2 = points.unsqueeze(1).repeat(1, n_points, 1).reshape(-1,2)
    dist_matrix = dist(p1, p2, mode).reshape(n_points, n_points)
    maps  = [map_row(relation_graph[i,:], dist_matrix[i,:],n) for i in range(n)]
    return np.sum(maps)/n