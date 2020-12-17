import tqdm
import click
from pathlib import Path

import cv2 as cv
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from bbox import BBox
from tracking import EmbeddingBuilder

EMB_CACHE_FILE_NAME = '_emb_cache.npy'

def mean_average_precision(pred, rank=10):
    n_vals = min(rank, pred.shape[1])
    corr_cum_sum = np.cumsum(pred, axis=1)
    vals_num_range = np.repeat(
        np.arange(1, n_vals + 1)[None, ...], repeats=pred.shape[0], axis=0)
    precisions_at_k = (corr_cum_sum / vals_num_range) * pred
    average_precisions_at_k = (np.sum(precisions_at_k, axis=1) /
                               np.sum(pred, axis=1))
    return np.mean(average_precisions_at_k)

def iterate_images_batch(gallery_dir_path, gallery_file_names):
    gallery_dir = Path(gallery_dir_path)
    
    for image_file_name in tqdm.tqdm(gallery_file_names):
        image = cv.imread(
            str(gallery_dir / image_file_name), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        yield image, BBox(0, 0, image.shape[1], image.shape[0])

def read_file_names_from_file(file_path):
    return Path(file_path).read_text().split()

def read_indices_from_file(file_path):
    return [
        [int(index) - 1 for index in indices.split()]
        for indices in Path(file_path).read_text().split('\n')]

@click.command()
@click.argument('config_file_path')
@click.argument('weights_file_path')
@click.argument('gallery_dir_path')
@click.argument('gallery_names_file_path')
@click.argument('query_names_file_path')
@click.argument('gt_indices_file_path')
@click.argument('junk_indices_file_path')
def main(
        config_file_path, weights_file_path, gallery_dir_path,
        gallery_names_file_path, query_names_file_path, gt_indices_file_path,
        junk_indices_file_path):
    gallery_file_names = read_file_names_from_file(gallery_names_file_path)
    query_file_names = read_file_names_from_file(query_names_file_path)
    gt_indices = read_indices_from_file(gt_indices_file_path)
    junk_indices = read_indices_from_file(junk_indices_file_path)
    gallery_file_name_indices = {
        file_name: i for i, file_name in enumerate(gallery_file_names)}
    query_indices = list(
        map(gallery_file_name_indices.__getitem__, query_file_names))
    assert len(query_indices) == len(gt_indices) == len(junk_indices)
    
    if Path(EMB_CACHE_FILE_NAME).exists():
        gallery_emb = np.load(EMB_CACHE_FILE_NAME)
    else:
        emb_builder = EmbeddingBuilder(config_file_path, weights_file_path)
        gallery_emb = []
        for image, box in iterate_images_batch(
                gallery_dir_path, gallery_file_names):
            gallery_emb.append(emb_builder.build(image, box))
        gallery_emb = np.array(gallery_emb)
        np.save(EMB_CACHE_FILE_NAME, gallery_emb)
    
    query_emb = gallery_emb[query_indices]
    dist_mat = euclidean_distances(query_emb, gallery_emb)
    
    pred = []
    for query_index, junk_indices in zip(query_indices, junk_indices):
        pass
    
    # dist_mat = euclidean_distances(gallery_emb)
    # pred = np.argsort(dist_mat, axis=1)[:-1, 1:]
    
    print(f'emb. shape: {gallery_emb.shape}')
    
    return 0


if __name__ == '__main__':
    import sys
    
    sys.exit(main())
