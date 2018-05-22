
def create_model(opt):
    model = None
    print(opt.model)
    assert(opt.dataset_mode == 'aligned')
    from .pix2pix_model import Pix2PixModel
    model = Pix2PixModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
