def test_cellpose_imports_without_error():
    import cellquant as cellpose
    from cellquant import models, core
    model = models.CellposeModel()



def test_gui_imports_without_error():
    from cellquant import gui


def test_gpu_check():
    #     from cellquant import models
    #     models.use_gpu()
    from cellquant import core
    core.use_gpu()


def test_model_dir():
    import numpy as np

    from cellquant import models
    model = models.CellposeModel(pretrained_model='cyto3')
    masks = model.eval(np.random.randn(224, 224))[0]
    assert masks.shape == (224, 224)
