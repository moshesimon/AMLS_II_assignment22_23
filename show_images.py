
def img_display(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    img = npimg.squeeze(0)
    return img