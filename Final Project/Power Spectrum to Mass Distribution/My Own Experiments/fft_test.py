import numpy as np
import matplotlib.pyplot as plt


N = 12
L = 12
dx = L/N
f = np.fft.fftfreq(N)
k = 2*np.pi*f
abs_k = np.abs(k[:len(k)//2+1])
print("f: ",f)
print("k: ",k)
print("abs_k:", abs_k)
# k = np.sqrt(2)*1.57079633
# k = 2.0943951

x = np.arange(0,N+dx,dx)
xx = np.array(np.meshgrid(x,x))
xx = np.einsum('kji->ijk',xx)

diag_dist = np.sum(xx,axis=2)



# plt.imshow(diag_sin)
# plt.show()



plt.ion()
fig, axes = plt.subplots(1,2)

for k in abs_k:
    print("k: ",k)
    diag_sin = np.sin(k*diag_dist)

    ft_diag_sin = np.fft.fftshift(np.fft.fftn(diag_sin))    
    im0 = axes[0].imshow(np.real(ft_diag_sin))
    im1 = axes[1].imshow(np.imag(ft_diag_sin))
    cb0 = fig.colorbar(im0,ax=axes[0])
    cb1 = fig.colorbar(im1,ax=axes[1])
    plt.pause(1)

    cb0.remove()
    cb1.remove()

