# class VelocityModel:
#     def __init__(self, position: np.ndarray, friction: float = 0.7) -> None:
#         assert 0 <= friction <= 1
#
#         self.position: np.ndarray = position.copy()
#         self.velocity: np.ndarray = np.array([0, 0])
#         self.friction: float = friction
#
#     def update(self, position: np.ndarray) -> None:
#         self.velocity = (self.friction * self.velocity +
#                          (1 - self.friction) * (position - self.position))
#         self.position = position


# class ObjectTemplate:
#     TEMPLATE_SIZE = (64, 64)
#     UPDATE_DECAY = 0.7
#
#     def __init__(self, image: np.ndarray, box: BBox) -> None:
#         self.template: np.ndarray = self.extract_resized_roi(image, box)
#
#     def update(self, image: np.ndarray, box: BBox) -> None:
#         roi = self.extract_resized_roi(image, box)
#         self.template = (self.template * (1 - self.UPDATE_DECAY) +
#                          roi * self.UPDATE_DECAY)
#
#     def calc_dist(self, other: 'ObjectTemplate') -> float:
#         # return 1 - ssim(self.template, other.template, multichannel=True)
#         return distance.cosine(
#             self.template.flatten(), other.template.flatten())
#
#     @staticmethod
#     def extract_resized_roi(image: np.ndarray, box: BBox) -> np.ndarray:
#         (x1, y1), (x2, y2) = box.top_left, box.bottom_right
#
#         x1 = np.clip(x1, 0, image.shape[1] - 1)
#         y1 = np.clip(y1, 0, image.shape[0] - 1)
#         x2 = np.clip(x2, 0, image.shape[1] - 1)
#         y2 = np.clip(y2, 0, image.shape[0] - 1)
#
#         roi = image[y1:y2, x1:x2]
#         roi = cv.resize(
#             roi, ObjectTemplate.TEMPLATE_SIZE, interpolation=cv.INTER_AREA)
#         return roi.astype(np.float)
