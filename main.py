from sampler import Sampler

if __name__ == '__main__':
    ########## Test CPPN ##########
    # sampler = Sampler(model_type='CPPN', z_dim=4, c_dim=3, scale=8.0, net_size=32)

    ########## Test different resolutions ##########
    # z1 = sampler.generate_z()
    # img = sampler.generate(z1, 256, 256)
    # sampler.save_png(img, 'img_256.png')
    # img = sampler.generate(z1, 512, 512)
    # sampler.save_png(img, 'img_512.png')

    ########## Generate several wallpapers ##########
    # for i in range(10):
    #     sampler.reinit()
    #     img = sampler.generate(None, 2880, 1800)
    #     sampler.save_png(img, 'img{}_2880_1800.png'.format(i + 1))

    ########## Test RPNN ##########
    sampler = Sampler(model_type='RPPN', z_dim=4, c_dim=3, scale=4.0, net_size=32)

    ########## Test different resolutions ##########
    # z1 = sampler.generate_z()
    # img1 = sampler.generate(z1, 256, 256)
    # sampler.save_png(img1, 'img_k3_256.png')
    # img2 = sampler.generate(z1, 512, 512)
    # sampler.save_png(img2, 'img_k3_512.png')

    ########## Test different number of repetitions ##########
    # z1 = sampler.generate_z()
    # img1 = sampler.generate(z1, 256, 256, k=0)
    # sampler.save_png(img1, 'img_k0_256.png')
    # img2 = sampler.generate(z1, 256, 256, k=1)
    # sampler.save_png(img2, 'img_k1_256.png')
    # img3 = sampler.generate(z1, 256, 256, k=2)
    # sampler.save_png(img3, 'img_k2_256.png')
    # img4 = sampler.generate(z1, 256, 256, k=3)
    # sampler.save_png(img4, 'img_k3_256.png')
    # img5 = sampler.generate(z1, 256, 256, k=4)
    # sampler.save_png(img5, 'img_k4_256.png')
    # img6 = sampler.generate(z1, 256, 256, k=5)
    # sampler.save_png(img6, 'img_k5_256.png')

    ########## Test gif with different number of repetitions ##########
    z1 = sampler.generate_z()
    sampler.save_anim_gif(z1, z1, 'anim_k0_k24_256.gif', n_frame=24, duration1=0.5, \
                     duration2=1.0, duration=0.2, x_dim=256, y_dim=256, scale=4.0, k1=0, k2=24, reverse=True)

    ########## Test mp4 with different number of repetitions ##########
    # z1 = sampler.generate_z()
    # sampler.save_anim_mp4(z1, z1, 'anim_k0_k24_256.mp4', n_frame=24, fps=12, x_dim=256, y_dim=256, scale=10.0, k1=0, k2=24)
