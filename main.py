from sampler import Sampler

if __name__ == '__main__':
    sampler = Sampler(z_dim=4, c_dim=3, scale=8.0, net_size=64)
    z1 = sampler.generate_z()
    img = sampler.generate(z1)
    sampler.show_image(img)