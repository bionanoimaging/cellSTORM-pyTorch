from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--roisize', type=int, default=int("-1"), help='Crop-parameter for video file')
        self.parser.add_argument('--xcenter', type=int, default=int("-1"), help='Crop-parameter for video file')
        self.parser.add_argument('--ycenter', type=int, default=int("-1"), help='Crop-parameter for video file')
        self.parser.add_argument('--padwidth', type=int, default=int("0"), help='How much padding added to the border?')
        
        self.isTrain = False
