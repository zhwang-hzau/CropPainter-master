from main import CropPainter as cp
''' cls: Panicle / Rice / Maize / Cotton
    run: train / test; 
        if you set run="train", please ignore <single_generate> <single_traits> <single_image_name>
    single_generate:
        if you set single_generate=True, please set <traits> carefully and set the <single_image_name> you want generated, ignore <traits_path>
        the generated images in output/models/<cls>_default/Model/iteration560000/single_samples/test/
    traits_path: train or test set (.csv) path '''
# Panicle;Cotton:18 traits
# Rice;Maize:16 traits, no YPA,YTR(first and third from the bottom of 18traits)
traits = [1.60124, 5.64309, 0.84947, 631.64387, 0.83806,
          0.81142, 5553, 113, 212, 0.2318,
          17236, 0.32217, 0.39546, 1.43473, 5131,
          422, 2196, 0.07599]  # Panicle
# traits = [0.42108, 2.34451, 0.701, 76.35607, 0.92789,
#           0.3814, 2610, 179, 188, 0.08801,
#           15727, 0.15324, 0.94108, 1.32056, 2618, 2268]  # Rice
cp(cls="Panicle",
   run="test",
   single_generate=True, single_traits=traits, single_image_name='test.png',
   traits_path='../data/Panicle/test/traits.csv')
