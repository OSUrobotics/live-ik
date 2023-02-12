#!/usr/bin/env python3

# Read in one masked image, the flow image, and the two rgbd images and
#  a) Find the most likely mask
#  b) Fit a bezier to that mask
#    b.1) use PCA to find the center estimate of the mask at 3 locations
#    b.2) Fit the bezier to the horizontal slices, assuming mask is correct
#    b.3) Fit the bezier to the edges

import numpy as np
from glob import glob
import csv
import cv2
import json
from os.path import exists
#from cyl_fit_2d import Quad
#from line_seg_2d import draw_line, draw_box, draw_cross, LineSeg2D
#from scipy.cluster.vq import kmeans, whiten, vq

class LeaderDetector:
    image_type = {"Mask", "Flow", "RGB1", "RGB2", "Edge", "RGB_Stats", "Mask_Stats", "Edge_debug"}

    _width = 0
    _height = 0

    _x_grid = None
    _y_grid = None

    @staticmethod
    def _init_grid_(in_im):
        """ INitialize width, height, xgrid, etc so we don't have to keep re-making it
        :param in_im: Input image
        """
        if LeaderDetector._width == in_im.shape[1] and LeaderDetector._height == in_im.shape[0]:
            return
        LeaderDetector._width = in_im.shape[1]
        LeaderDetector._height = in_im.shape[0]

        LeaderDetector._x_grid, LeaderDetector._y_grid = np.meshgrid(np.linspace(0.5, LeaderDetector._width - 0.5, LeaderDetector._width), np.linspace(0.5,  LeaderDetector._height -  0.5,  LeaderDetector._height))

    def __init__(self, path, image_name, b_output_debug=True, b_recalc=False):
        """ Read in the image, mask image, flow image, 2 rgb images
        @param path: Directory where files are located
        @param image_name: image number/name as a string
        @param b_recalc: Force recalculate the result, y/n"""

        self.path_debug = path + "DebugImages/"
        path_calculated = path + "CalculatedData/"

        self.name = image_name
        # Read in all images that have name_ and are not debugging images
        self.images = self.read_images(path, image_name)
        # Split the mask into connected components, each of which might be a vertical leader
        self.vertical_leader_masks = self.split_mask(self.images["Mask"], b_one_mask=True, b_debug=b_output_debug)
        self.vertical_leader_stats = []
        self.vertical_leader_quads = []

        # For each component of the mask image, calculate or read in statistics (center, eigen vectors)
        print("Calculating stats")
        for i, mask in enumerate(self.vertical_leader_masks):
            fname_stats = path_calculated + self.name + f"_mask_{i}.json"
            if b_recalc or not exists(fname_stats):
                stats_dict = self.stats_image(self.images["Mask"], mask)
                for k, v in stats_dict.items():
                    try:
                        if v.size == 2:
                            stats_dict[k] = [v[0], v[1]]
                    except:
                        pass
                # If this fails, make a CalculatedData and DebugImages folder in the data/forcindy folder
                with open(fname_stats, 'w') as f:
                    json.dump(stats_dict, f)
            elif exists(fname_stats):
                with open(fname_stats, 'r') as f:
                    stats_dict = json.load(f)

            for k, v in stats_dict.items():
                try:
                    if len(v) == 2:
                        stats_dict[k] = np.array([v[0], v[1]])
                except:
                    pass
            self.vertical_leader_stats.append(stats_dict)

        # For each of the masks, see if we have reasonable stats
        #   Save points in debug image
        if b_output_debug:
            self.images["Mask_Stats"] = np.copy(self.images["Mask"])
            self.images["RGB_Stats"] = np.copy(self.images["RGB1"])
            for i, stats in enumerate(self.vertical_leader_stats):
                self.images["Mask_Stats"] = self.images["Mask"] / 2
                try:
                    p1 = stats["lower_left"]
                    p2 = stats["upper_right"]
                    self.images["Mask_Stats"][self.vertical_leader_masks[i]] = 255
                    draw_line(self.images["RGB_Stats"], p1, p2, (128, 128, 128), 2)
                    draw_line(self.images["Mask_Stats"], p1, p2, (128, 128, 128), 1)

                    pc = ["center"]
                    draw_cross(self.images["RGB_Stats"], pc, (128, 128, 128), 1, 2)
                    draw_cross(self.images["Mask_Stats"], pc, (180, 180, 128), 1, 3)

                except:
                    pass

                cv2.imwrite(self.path_debug + image_name + "_" + f"{i}_mask_points.png", self.images["Mask_Stats"])
            # cv2.imwrite(self.path_debug + image_name + "_" + "rgb_points.png", self.images["RGB_Stats"])

        # Fit a quad to each vertical leader
        print("Fitting quads")
        for i, stats in enumerate(self.vertical_leader_stats):
            print(f"  {image_name}, mask {i}")
            image_mask = np.zeros(self.images["Mask"].shape, dtype=self.images["Mask"].dtype)
            fname_quad = path_calculated + self.name + "_" + image_name + f"_{i}_quad.json"
            fname_params = path_calculated + self.name + "_" + image_name + f"_{i}_quad_params.json"
            quad = None
            if exists(fname_quad) and not b_recalc:
                quad = Quad([0, 0], [1,1], 1)
                quad.read_json(fname_quad)
                with open(fname_params, 'r') as f:
                    params = json.load(f)
            else:
                image_mask[self.vertical_leader_masks[i]] = 255
                quad, params = self.fit_quad(image_mask, pts=stats, b_output_debug=b_output_debug, quad_name=i)
                quad.write_json(fname_quad)
                with open(fname_params, 'w') as f:
                    json.dump(params, f)
            self.vertical_leader_quads.append(quad)

            if b_output_debug:
                # Draw the edge and original image with the fitted quad and rects
                im_covert_back = cv2.cvtColor(self.images["Edge"], cv2.COLOR_GRAY2RGB)
                im_orig_debug = np.copy(self.images["RGB0"])

                # Draw the original, the edges, and the depth mask with the fitted quad
                quad.draw_quad(im_orig_debug)
                if quad.is_wire():
                    draw_cross(im_orig_debug, quad.p0, (255, 0, 0), thickness=2, length=10)
                    draw_cross(im_orig_debug, quad.p2, (255, 0, 0), thickness=2, length=10)
                else:
                    quad.draw_boundary(im_orig_debug, 10)
                    quad.draw_edge_rects(im_covert_back, step_size=params["step_size"], perc_width=params["width"])

                im_both = np.hstack([im_orig_debug, im_covert_back])
                cv2.imshow("Original and edge and depth", im_both)
                cv2.imwrite(self.path_debug + self.name + "_" + image_name + f"_{i}_quad.png", im_both)

                quad.draw_quad(self.images["RGB_Stats"])

        for quad in self.vertical_leader_quads:
            score = self.score_quad(quad)
            print(f"Score {score}")
    

    def read_images(self, path, image_name):
        """ Read in all of the mask, rgb, flow images
        If Edge image does not exist, create it
        @param path: Directory where files are located
        @param image_name: image number/name as a string
        @returns dictionary of images with image_type as keywords """

        images = {}
        search_path = "./test.png"#f"{path}{image_name}_*.png"
        print(search_path)
        fnames = glob(search_path)
        if fnames is None:
            raise ValueError(f"No files in directory {search_path}")

        for n in fnames:
            if "mask" in n:
                im_mask_color = cv2.imread(n)
                images["Mask"] = cv2.cvtColor(im_mask_color, cv2.COLOR_BGR2GRAY)
            elif "rgb0" in n:
                images["RGB0"] = cv2.imread(n)
            elif "rgb1" in n:
                images["RGB1"] = cv2.imread(n)
            elif "flow" in n:
                images["Flow"] = cv2.imread(n)
            elif "edge" in n:
                im_edge_color = cv2.imread(n)
                images["Edge"] = cv2.cvtColor(im_edge_color, cv2.COLOR_BGR2GRAY)
            else:
                print(f" Skipping {n}")

        if "Edge" not in images:
            im_gray = cv2.cvtColor(images["RGB0"], cv2.COLOR_BGR2GRAY)
            images["Edge"] = cv2.Canny(im_gray, 50, 150, apertureSize=3)
            cv2.imwrite(path + image_name + "_edge.png", images["Edge"])

        return images

    def split_mask(self, in_im_mask, b_one_mask=True, b_debug=False):
        """Split the mask image up into connected components, discarding anything really small
        @param in_im_mask - the mask image
        @param b_debug - print out mask labeled image
        @return a list of boolean indices for each component"""
        output = cv2.connectedComponentsWithStats(in_im_mask)
        labels = output[1]
        stats = output[2]

        ret_masks = []
        i_widest = 0
        i_area = 0
        for i, stat in enumerate(stats):
            if np.sum(in_im_mask[labels == i]) == 0:
                continue

            if stat[cv2.CC_STAT_WIDTH] < 5:
                continue
            if stat[cv2.CC_STAT_HEIGHT] < 0.5 * in_im_mask.shape[1]:
                continue
            if i_area < stat[cv2.CC_STAT_AREA]:
                i_widest = len(ret_masks)
                i_area = stat[cv2.CC_STAT_AREA]
            ret_masks.append(labels == i)

        if b_debug:
            labels = 128 + labels * (120 // output[0])
            cv2.imwrite(self.path_debug + self.name + "_" + "labels.png", labels)

        try:
            if b_one_mask:
                return [ret_masks[i_widest]]
        except:
            pass
        return ret_masks

    def stats_image(self, in_im, pixs_in_mask):
        """ Add statistics (bounding box, left right, orientation, radius] to image
        Note: Could probably do this without transposing image, but...
        @param im image
        @returns stats as a dictionary of values"""

        LeaderDetector._init_grid_(in_im)

        xs = LeaderDetector._x_grid[pixs_in_mask]
        ys = LeaderDetector._y_grid[pixs_in_mask]

        stats = {}
        stats["x_min"] = np.min(xs)
        stats["y_min"] = np.min(ys)
        stats["x_max"] = np.max(xs)
        stats["y_max"] = np.max(ys)
        stats["x_span"] = stats["x_max"] - stats["x_min"]
        stats["y_span"] = stats["y_max"] - stats["y_min"]

        avg_width = 0.0
        count_width = 0
        if stats["x_span"] > stats["y_span"]:
            stats["Direction"] = "left_right"
            stats["Length"] = stats["x_span"]
            for r in range(0, LeaderDetector._width):
                if sum(pixs_in_mask[:, r]) > 0:
                    avg_width += sum(pixs_in_mask[:, r] > 0)
                    count_width += 1
        else:
            stats["Direction"] = "up_down"
            stats["Length"] = stats["y_span"]
            for c in range(0, LeaderDetector._height):
                if sum(pixs_in_mask[c, :]) > 0:
                    avg_width += sum(pixs_in_mask[c, :] > 0)
                    count_width += 1
        stats["width"] = avg_width / count_width
        stats["center"] = np.array([np.mean(xs), np.mean(ys)])

        x_matrix = np.zeros([2, xs.shape[0]])
        x_matrix[0, :] = xs.transpose() - stats["center"][0]
        x_matrix[1, :] = ys.transpose() - stats["center"][1]
        covariance_matrix = np.cov(x_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        if eigen_values[0] < eigen_values[1]:
            stats["EigenValues"] = [np.min(eigen_values), np.max(eigen_values)]
            stats["EigenVector"] = eigen_vectors[1, :]
        else:
            stats["EigenValues"] = [np.min(eigen_values), np.max(eigen_values)]
            stats["EigenVector"] = eigen_vectors[0, :]
        eigen_ratio = stats["EigenValues"][1] / stats["EigenValues"][0]
        stats["EigenVector"][1] *= -1
        stats["EigenRatio"] = eigen_ratio
        stats["lower_left"] = stats["center"] - stats["EigenVector"] * (stats["Length"] * 0.5)
        stats["upper_right"] = stats["center"] + stats["EigenVector"] * (stats["Length"] * 0.5)
        print(stats)
        print(f"Eigen ratio {eigen_ratio}")
        return stats

    def fit_quad(self, im_mask, pts, b_output_debug=True, quad_name=0):
        """ Fit a quad to the mask, edge image
        @param im_mask - the image mask
        @param pts - the stats from the stats call
        @param b_output_debug - output mask with quad at the intermediate step
        @returns fitted quad"""

        # Fit a quad to the trunk
        pt_lower_left = pts['center']
        vec_len = pts["Length"] * 0.4
        while pt_lower_left[0] > 2 + pts['x_min'] and pt_lower_left[1] > 2 + pts['y_min']:
            pt_lower_left = pts["center"] - pts["EigenVector"] * vec_len
            vec_len = vec_len * 1.1

        pt_upper_right = pts['center']
        vec_len = pts["Length"] * 0.4
        while pt_upper_right[0] < -2 + pts['x_max'] and pt_upper_right[1] < -2 + pts['y_max']:
            pt_upper_right = pts["center"] + pts["EigenVector"] * vec_len
            vec_len = vec_len * 1.1

        quad = Quad(pt_lower_left, pt_upper_right, 0.5 * pts['width'])

        # Current parameters for the vertical leader
        params = {"step_size": int(quad.radius_2d * 1.5), "width_mask": 1.4, "width": 0.25}

        # Iteratively move the quad to the center of the mask
        for i in range(0, 5):
            res = quad.adjust_quad_by_mask(im_mask,
                                           step_size=params["step_size"], perc_width=params["width_mask"],
                                           axs=None)
            print(f"Res {res}")

        if b_output_debug:
            im_debug = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2RGB)
            quad.draw_quad(im_debug)
            quad.draw_boundary(im_debug)
            cv2.imwrite(self.path_debug + self.name + "_" + self.name + f"_{quad_name}_quad_fit_mask.png", im_debug)

        # Now do the hough transform - first draw the hough transform edges
        for i in range(0, 5):
            ret = quad.adjust_quad_by_hough_edges(self.images["Edge"], step_size=params["step_size"], perc_width=params["width"], axs=None)
            print(f"Res Hough {ret}")

        return quad, params

    def adjust_quad_by_flow(self):
        """ Not really useful now - fixes the mask by calculating the average depth in the flow mask under the bezier
        the trimming off mask pixels that don't belong"""
        # Use the flow image to make a better mask
        print("Quad in flow mask")
        self.flow_masks_images = []
        self.flow_masks_labels = []
        self.flow_quad = []
        for i, quad in enumerate(self.vertical_leader_quads):
            print(f"  {image_name} {i}")

            fname_quad_flow_mask = path_calculated + self.name + "_" + image_name + f"_{i}_quad_flow_mask.png"
            im_flow_mask = None
            im_flow_mask_labels = None
            if exists(fname_quad_flow_mask) and not b_recalc:
                im_flow_mask = cv2.cvtColor(cv2.imread(fname_quad_flow_mask), cv2.COLOR_BGR2GRAY)
            else:
                im_flow_mask, im_flow_mask_labels = self.flow_mask(quad)
                if b_output_debug:
                    cv2.imwrite(self.path_debug + self.name + "_" + image_name + f"_{i}_quad_flow_labels.png", im_flow_mask_labels)
                cv2.imwrite(fname_quad_flow_mask, im_flow_mask)

            fname_quad_flow = path_calculated + self.name + "_" + image_name + f"_{i}_quad_flow.json"
            fname_params_flow = path_calculated + self.name + "_" + image_name + f"_{i}_quad_params_flow.json"
            quad_flow = None
            if exists(fname_quad_flow) and not b_recalc:
                quad_flow = Quad([0, 0], [1,1], 1)
                quad_flow.read_json(fname_quad_flow)
                with open(fname_params_flow, 'r') as f:
                    params = json.load(f)
            else:
                quad_flow, params = self.fit_quad_flow(im_flow_mask)
                quad_flow.write_json(fname_quad_flow)
                with open(fname_params_flow, 'w') as f:
                    json.dump(params, f)

            self.flow_masks_images.append(im_flow_mask)
            self.flow_masks_labels.append(params)
            self.flow_quad.append(quad_flow)

            if b_output_debug:
                # Draw the edge and original image with the fitted quad and rects
                im_covert_back = cv2.cvtColor(self.images["Edge"], cv2.COLOR_GRAY2RGB)
                im_orig_debug = np.copy(self.images["RGB0"])

                # Draw the original, the edges, and the depth mask with the fitted quad
                quad_flow.draw_quad(im_orig_debug)
                if quad_flow.is_wire():
                    draw_cross(im_orig_debug, quad_flow.p0, (255, 0, 0), thickness=2, length=10)
                    draw_cross(im_orig_debug, quad_flow.p2, (255, 0, 0), thickness=2, length=10)
                else:
                    quad_flow.draw_boundary(im_orig_debug, 10)
                    quad_flow.draw_edge_rects(im_covert_back, step_size=params["step_size"], perc_width=params["width"])

                im_both = np.hstack([im_orig_debug, im_covert_back])
                cv2.imwrite(self.path_debug + self.name + "_" + image_name + f"_{i}_quad_flow.png", im_both)

                quad_flow.draw_quad(self.images["RGB_Stats"])

    def fit_quad_flow(self, im_flow_mask, quad, b_output_debug=True):
        """ Fit a quad to the mask, edge image
        @param im_flow_mask - the flow image
        @param quad - the quad
        @param b_output_debug - output mask with quad at the intermediate step
        @returns fitted quad"""

        # Fit a quad to the trunk
        quad = Quad(quad.p0, quad.p2, quad.radius_2d, mid_pt=quad.p1)

        # Current parameters for the vertical leader
        params = {"step_size": int(quad.radius_2d * 1.5), "width_mask": 1.4, "width": 0.3}

        # Iteratively move the quad to the center of the mask
        for i in range(0, 5):
            res = quad.adjust_quad_by_mask(im_flow_mask,
                                           step_size=params["step_size"], perc_width=params["width_mask"],
                                           axs=None)
            print(f"Res {res}")

        if b_output_debug:
            im_debug = cv2.cvtColor(im_flow_mask, cv2.COLOR_GRAY2RGB)
            quad.draw_quad(im_debug)
            quad.draw_boundary(im_debug)
            cv2.imwrite(self.path_debug + self.name + "_" + self.name + f"_{i}_quad_fit_mask_flow.png", im_debug)

        # Now do the hough transform - first draw the hough transform edges
        for i in range(0, 5):
            ret = quad.adjust_quad_by_hough_edges(self.images["Edge"], step_size=params["step_size"], perc_width=params["width"], axs=None)
            print(f"Res Hough {ret}")

        return quad, params

    def flow_mask(self, quad):
        """ Use the fitted quad and the original mask to extract a better mask from the flow image
        @param quad - the quad we've fitted so far
        @return im_mask - a better image mask"""
        im_flow = self.images["Flow"]

        im_inside = quad.interior_rects_mask((im_flow.shape[0], im_flow.shape[1]), step_size=30, perc_width=0.5)
        im_inside = im_inside.reshape((im_flow.shape[0] * im_flow.shape[1]))
        im_flow_reshape = im_flow.reshape((im_flow.shape[0] * im_flow.shape[1], 3))
        n_inside = np.count_nonzero(im_inside)
        n_total = im_flow.shape[0] * im_flow.shape[1]
        im_flow_whiten = whiten(im_flow_reshape)
        color_centers = kmeans(im_flow_whiten, 4)

        pixel_labels = vq(im_flow_whiten, color_centers[0])
        label_count = [(np.count_nonzero(np.logical_and(pixel_labels[0] == i, im_inside == True)), i) for i in range(0, 4)]
        label_count.sort()

        im_mask_labels = np.zeros(im_inside.shape, dtype=im_flow.dtype)
        im_mask = np.zeros(im_inside.shape, dtype=im_flow.dtype)
        n_div = 125 // 3
        for i, label in enumerate(label_count):
            im_mask_labels[np.logical_and(pixel_labels[0] == label[1], im_inside == True)] = 125 + int(i * n_div)
            im_mask_labels[np.logical_and(pixel_labels[0] == label[1], im_inside == False)] = int(i * n_div)
        im_mask[pixel_labels[0] == label_count[-1][1]] = 255
        return im_mask.reshape((im_flow.shape[0], im_flow.shape[1])), im_mask_labels.reshape((im_flow.shape[0], im_flow.shape[1]))

    def score_quad(self, quad):
        """ See if the quad makes sense over the optical flow image
        @quad - the quad
        """

        # Two checks: one, are the depth/optical fow values largely consistent under the quad center
        #  Are there boundaries in the optical flow image where the edge of the quad is?
        im_flow_mask = cv2.cvtColor(self.images["Flow"], cv2.COLOR_BGR2GRAY)
        perc_consistant, stats_slice = quad.check_interior_depth(im_flow_mask)

        diff = 0
        for i in range(1, len(stats_slice)):
            diff_slices = np.abs(stats_slice[i]["Median"] - stats_slice[i-1]["Median"])
            if diff_slices > 20:
                print(f"Warning: Depth values not consistant from slice {self.name} {i} {stats_slice}")
            diff += diff_slices
        if perc_consistant < 0.9:
            print(f"Warning: not consistant {self.name} {stats_slice}")
        return perc_consistant, diff / (len(stats_slice) - 1)


if __name__ == '__main__':
    """
    image = cv2.imread('test.jpg')
    leader = LeaderDetector()
    leader._init_grid_(image)
    """
    path = "./data/predictions/"
    #path = "./forcindy/"
    bp = LeaderDetector(path, "test", b_output_debug=True, b_recalc=True)
    