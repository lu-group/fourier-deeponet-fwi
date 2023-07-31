import torch
import numpy as np
import os
path = './seismic/f'
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ricker(f, dt, nt):
    nw = 2.2 / f / dt
    nw = 2 * np.floor(nw / 2) + 1
    nc = np.floor(nw / 2)
    k = np.arange(nw)

    alpha = (nc - k) * f * dt * np.pi
    beta = alpha ** 2
    w0 = (1 - beta * 2) * np.exp(-beta)
    w = np.zeros(nt)
    w[:len(w0)] = w0
    return w


def get_Abc(vp, nbc, dx):
    dimrange = 1.0 * torch.unsqueeze(torch.arange(nbc, device=vp.get_device()), dim=-1)
    damp = torch.zeros_like(vp, device=vp.get_device(), requires_grad=False)

    velmin, _ = torch.min(vp.view(vp.shape[0], -1), dim=-1, keepdim=False)

    nzbc, nxbc = vp.shape[2], vp.shape[3]
    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc
    a = (nbc - 1) * dx

    kappa = 3.0 * velmin * np.log(1e7) / (2.0 * a)
    kappa = torch.unsqueeze(kappa, dim=0)
    kappa = torch.repeat_interleave(kappa, nbc, dim=0).to(vp.get_device())

    damp1d = kappa * (dimrange * dx / a) ** 2
    damp1d = damp1d.permute(1, 0).unsqueeze(1)
    damp[:, :, :nbc, :] = torch.repeat_interleave(torch.flip(damp1d, dims=[-1]).unsqueeze(-1), vp.shape[-1], dim=-1)
    damp[:, :, -nbc:, :] = torch.repeat_interleave(damp1d.unsqueeze(-1), vp.shape[-1], dim=-1)
    damp[:, :, :, :nbc] = torch.repeat_interleave(torch.flip(damp1d, dims=[-1]).unsqueeze(-2), vp.shape[-2], dim=-2)
    damp[:, :, :, -nbc:] = torch.repeat_interleave(damp1d.unsqueeze(-2), vp.shape[-2], dim=-2)
    return damp


def adj_sr(sx, sz, gx, gz, dx, nbc):
    isx = np.around(sx / dx) + nbc
    isz = np.around(sz / dx) + nbc

    igx = np.around(gx / dx) + nbc
    igz = np.around(gz / dx) + nbc
    return isx.astype('int'), int(isz), igx.astype('int'), int(igz)


def FWM(v, nbc=120, dx=10, nt=1000, dt=1e-3, f=15, sx=np.linspace(0, 69, 5)*10, sz=10,
        gx=np.linspace(0, 69, 70)*10, gz=10, sampling_rate=1):

    src=ricker(f,dt, nt)
    alpha = (v*dt/dx) ** 2
    abc = get_Abc(v, nbc, dx)
    kappa = abc*dt

    c1 = -2.5
    c2 = 4.0/3.0
    c3 = -1.0/12.0

    temp1 = 2+2*c1*alpha-kappa
    temp2 = 1-kappa
    beta_dt = (v*dt) ** 2

    ns = len(sx)
    isx,isz,igx,igz = adj_sr(sx,sz,gx,gz,dx,nbc)
    seis = []
    p1 = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=v.get_device(), requires_grad=True)
    p0 = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=v.get_device(), requires_grad=True)
    p  = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=v.get_device(), requires_grad=True)
    for i in range(nt):
        p = (temp1*p1 - temp2*p0 + alpha *
        (c2*(torch.roll(p1, 1, dims = -2) + torch.roll(p1, -1, dims = -2) + torch.roll(p1, 1, dims = -1)+ torch.roll(p1, -1, dims = -1))
        +c3*(torch.roll(p1, 2, dims = -2) + torch.roll(p1, -2, dims = -2) + torch.roll(p1, 2, dims = -1)+ torch.roll(p1, -2, dims = -1))
        ))
        for loc in range(ns):
            p[:,loc,isz,isx[loc]] = p[:,loc,isz,isx[loc]] + beta_dt[:,0,isz,isx[loc]] * src[i]
        if i % sampling_rate == 0:
            seis.append(torch.unsqueeze(p[:, :, [igz]*len(igx), igx], dim=2))
        p0=p1
        p1=p
    return torch.cat(seis, dim=2)


if __name__ == "__main__":
    ####################################
    # Foward Modeling
    ####################################

    grids = 70
    # time interval
    nt = 1000
    # grid
    dx = 10
    # bc
    nbc = 120
    # grid t
    dt = 1e-3
    # src positions
    sz = 10
    sx = np.linspace(0, grids-1, num = 5)*dx
    # receivers positions
    gx = np.linspace(0, grids - 1, num=grids) * dx
    gz = 10


    #######################################
    # Velocity Generation Step 1
    #######################################
    num_samples = 500
    for iii in range(134):
        print(iii)
        f_list = np.load(f"./f/f{iii+1}.npy").astype(np.float32).reshape(num_samples, 1)
        vp = np.load(f"./velocity/model{iii+1}.npy").astype(np.float32)
        vp = np.pad(vp, ((0,0), (0,0), (nbc,nbc), (nbc,nbc)), 'edge')
        v_torch = torch.from_numpy(vp).to(device)
        seis = []
        for i in range(num_samples):
            f = f_list[i]
            v_torch_i = v_torch[i:i+1]
            with torch.no_grad():
                seis_i = FWM(v_torch_i,nbc,dx,nt,dt,f,sx,sz,gx,gz).cpu().detach().numpy()
            seis.append(seis_i)
        seis = np.concatenate(seis)
        np.save(f'./seismic/loc/data{iii+1}.npy',seis)
        torch.cuda.empty_cache()

