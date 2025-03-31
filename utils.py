from tqdm import tqdm
from scipy.signal import medfilt
from fitps import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

import warnings
import numpy as np
import itertools as it


def read_signatures(reader, ids=None):
    if ids is None:
        ids = range(len(reader))
    else:
        ids = ids.tolist()

    signatures = []

    for n in tqdm(ids):
        v, i, *args = reader[n]
        v = medmeanfilt(v, window_size=7)
        i = medmeanfilt(i, window_size=7)
        signatures.append((v, i, *args))

    return signatures


def active_power(v, i):
    return (v * i).mean()


def get_running_interval(i, I_on=0.05, I_min=0.1):
    assert len(i.shape) == 2
    I = abs(i).max(1)

    s = (I > I_min).astype(int)
    ds = np.diff(s, prepend=s[0])
    on = (ds > 0).nonzero()[0]
    off = (ds < 0).nonzero()[0]

    if on.size == 0:
        return None

    if off.size == 0:
        off = np.append(off, len(s))

    if off[0] < on[0]:
        on = np.insert(on, 0, 0)

    if off[-1] < on[-1]:
        off = np.append(off, len(s))

    assert len(on) == len(off)

    locs = np.stack((on, off)).T

    # print(len(I))
    # print(on)
    # print(off)
    assert (locs[:, 1] > locs[:, 0]).all()

    for k, (a, b) in enumerate(locs):
        ia = i[a]
        on = (abs(np.diff(ia)) > I_on).nonzero()[0]

        if on.size > 0:
            on = on[0]
        else:
            on = 0

        if on > 0 and abs(ia[:on]).mean() > I_min:
            on = 0

        ib = i[b - 1]
        off = (abs(np.diff(ib[::-1])) > I_on).nonzero()[0]

        if off.size > 0:
            off = off[0]
        else:
            off = 0

        if off > 0 and abs(ib[:off]).mean() > I_min:
            off = 0

        off = len(i[a:b].ravel()) - off - 1

        locs[k] = [a * i.shape[1] + on, a * i.shape[1] + off]

    return locs


def split_if_more(recordings, n_cycles):
    data = []

    for v, i, fs, f0, app, *other in recordings:
        agg_fmt = True
        if isinstance(app, str):
            app = [app]
            agg_fmt = False

        if len(other) > 0:
            locs = other[0]
        else:
            locs = None

        for j in range(0, len(v), n_cycles):
            vj = v[j:j + n_cycles]
            ij = i[j:j + n_cycles]

            if len(vj) == n_cycles:
                if locs is not None:
                    _locs = []
                    _devices = []

                    for device, (on, off) in zip(app, locs):
                        on = max(on - j * n_cycles * i.shape[1], 0)
                        off = max(
                            min(off - on // i.shape[1] * i.shape[1],
                                n_cycles * i.shape[1] - 1), 0)

                        if off == on:
                            continue
                        _devices.append(device)
                        _locs.append([on, off])

                    if not agg_fmt:
                        _devices = _devices[0]

                    data.append((vj, ij, fs, f0, _devices, _locs))
                else:
                    if not agg_fmt:
                        app = app[0]
                    data.append((vj, ij, fs, f0, app))

    return data


def split_by_ncomponents(recordings):
    data = []

    import matplotlib.pyplot as plt
    for v, i, fs, f0, apps, locs in tqdm(recordings):
        assert len(apps) == len(locs)

        Q = np.zeros((len(apps), len(v)), dtype=int)

        for j, (a, b) in enumerate(locs):
            a = a // i.shape[1]
            b = b // i.shape[1] + 1
            Q[j, a:b] += 1

        n_components = Q.sum(0)
        dn = np.diff(n_components, prepend=0, append=0)
        chpts = (dn != 0).nonzero()[0].ravel()

        a = chpts[0]

        for b in chpts[1:]:
            ids = Q[:, a:b].sum(1).nonzero()[0]
            assert len(ids) > 0

            _locs = []
            _devices = []

            for j in ids:
                device = apps[j]
                on, off = locs[j]

                on = max(0, on - a * v.shape[1])
                off = max(0, min((b - a) * v.shape[1] - 1,
                                 off - a * v.shape[1]))
                assert off >= on

                if off == on:
                    continue

                _devices.append(device)
                _locs.append([on, off])

            data.append((v[a:b], i[a:b], fs, f0, _devices, _locs))
            a = b

    return data


def meanfilt(x, window_size=7):
    x = np.convolve(x, np.ones(window_size) / window_size, 'valid')

    return x


def medmeanfilt(x, window_size=7):
    x = medfilt(x, window_size)
    x = meanfilt(x, window_size=window_size)

    return x


def nearest_f0(recordings):
    f0_list = list({f0 for _, _, _, f0, *_ in recordings})
    f0 = int(round(np.mean(f0_list)))

    return f0


def sync_recordings(
    recordings,
    window_size=1,
    outlier_thresh=0.1,
    zero_thresh=1e-4,
    f0_ref=None,
    progress_bar=True,
):
    fitps = FITPS(window_size=window_size,
                  outlier_thresh=outlier_thresh,
                  zero_thresh=zero_thresh)
    sync_recordings = []

    if f0_ref is None:
        f0_ref = nearest_f0(recordings)

    for rec in tqdm(recordings, disable=not progress_bar):
        v, i, fs, _, appliances, locs = rec

        try:
            fitps.fit(v)
        except OutliersDetected:
            continue
        else:
            T = math.ceil(fs / f0_ref)
            v = fitps.transform(v, cycle_size=T)
            i = fitps.transform(i, cycle_size=T, locs=locs)

            if isinstance(i, tuple):
                i, locs = i

            sync_recordings.append((v, i, fs, f0_ref, appliances, locs))

    dn = len(recordings) - len(sync_recordings)

    if dn > 0:
        warnings.warn(f'{dn} outliers were omitted.')

    return sync_recordings


def get_submetered(aggregated):
    assert not isinstance(aggregated[0][4], str | np.str_)

    submetered = [(v, i, fs, f0, apps[0], *other)
                  for v, i, fs, f0, apps, *other in aggregated
                  if len(apps) == 1]

    return submetered


def get_aggregated(aggregated):
    assert not isinstance(aggregated[0][4], str | np.str_)

    aggregated = [(v, i, fs, f0, apps, *other)
                  for v, i, fs, f0, apps, *other in aggregated
                  if len(apps) > 1]

    return aggregated


def filter_aggregated(D_agg, D_sub):
    devices_agg = []
    devices_sub = []

    for _, _, _, _, devices, _ in D_agg:
        if isinstance(devices, str):
            devices = [devices]

        devices_agg.extend(devices)

    devices_agg = set(devices_agg)

    for _, _, _, _, devices, _ in D_sub:
        if isinstance(devices, str):
            devices = [devices]

        devices_sub.extend(devices)

    devices_sub = set(devices_sub)
    missing_devices = devices_agg - devices_sub
    common_devices = devices_sub - missing_devices

    mask = []

    for _, _, _, _, devices, _ in D_agg:
        assert not isinstance(devices, str)
        m = True

        for device in devices:
            if device not in common_devices:
                m = False
                break

        mask.append(m)

    D_agg = filter_with_mask(D_agg, mask)

    return D_agg


def is_activation(device, thresh=0.1):
    _, i, _, _, _, *other = device

    if len(other) > 0:
        locs = np.asarray(other[0])

        if locs.min() > 0:
            return True

    _E0 = 1e-9

    u, l = i.max(1), i.min(1)
    u = u / (u.max() + _E0)
    # l = l / (l.max() + _E0)
    l = l / (l.min() + _E0)
    scores = [abs(u.max() - u.min())]
    scores += [abs(l.max() - l.min())]
    scores += [abs(u.max() - l.min())]
    scores += [abs(l.max() - u.min())]
    score = max(scores)
    is_activation = score > thresh

    return is_activation


def get_transients(data, thresh=0.1):
    return list(filter(lambda x: is_activation(x, thresh=thresh), data))


def get_steady_states(data, thresh=0.1):
    return list(filter(lambda x: not is_activation(x, thresh=thresh), data))


def tupleify_data(D):
    Dt = []

    for sample in D:
        v = sample['total']['v']
        i = sample['total']['i']
        fs = sample['meta']['fs']
        f0 = sample['meta']['f0']
        devices = [device['device'] for device in sample['devices']]
        locs = [device.get('locs', None) for device in sample['devices']]

        if None not in locs:
            locs = list(it.chain(*locs))
            assert len(locs) == len(devices)

        Dt.append((v, i, fs, f0, devices, locs))

    return Dt


def get_total_current(D, mask=None, normalize=False):
    I = []

    for _, i, _, _, _, *_ in D:
        I.append(i)

    I = np.asarray(I)

    if mask is not None:
        assert len(D) == len(mask)
        I = I[mask]

    if normalize:
        I = I / abs(I).max((1, 2), keepdims=True)

    return I


def get_total_power(D, mask=None, normalize=False):
    S = []

    for v, i, _, _, _, *_ in D:
        S.append((v * i).ravel())

    S = np.asarray(S)

    if mask is not None:
        assert len(D) == len(mask)
        S = S[mask]

    if normalize:
        S = S / abs(S).max(-1, keepdims=True)

    return S


def get_component_mask(D, k=1):
    mask = []

    for _, _, _, _, devices, *_ in D:
        assert not isinstance(devices, str | np.str_)

        mask.append(len(devices) == k)

    mask = np.asarray(mask)

    return mask


def get_device_names(D, mask=None):
    labels = []

    if mask is None:
        mask = np.ones(len(D), dtype=bool)
    else:
        assert len(D) == len(mask)

    for (_, _, _, _, devices, *_), take in zip(D, mask):
        # assert not isinstance(devices, str | np.str_)

        if take:
            labels.append(devices)

    return labels


def get_similar_devices(Xa, Xb, metric='cosine'):
    knn = NearestNeighbors(metric=metric)
    knn.fit(Xb)
    d, ids = knn.kneighbors(Xa, n_neighbors=1)

    return d.ravel(), ids.ravel()


def component_count(D):
    components = set()

    for _, _, _, _, devices, *_ in D:
        components.add(len(devices))

    return list(sorted(components))


def filter_with_mask(D, mask):
    return [data for data, take in zip(D, mask) if take]


def get_group_ids(D):
    I = defaultdict(list)

    for i, v in enumerate(D):
        v = tuple(sorted(v))
        I[v].append(i)

    return list(I.items())


def filter_with_devices(D, devices, strict=False):
    if strict:
        return [data for data in D if set(data[4]) == set(devices)]

    return [data for data in D if data[4] in set(devices)]


def filter_with_ids(D, ids):
    return [data for i, data in enumerate(D) if i in ids]


def get_low_power_mask(D, P_min=10):
    return [active_power(v, i) < P_min for v, i, _, _, _, *_ in D]


def rename_device(D, old_name, new_name):
    new_data = []

    for v, i, fs, f0, device, *other in D:
        if device == old_name:
            device = new_name

        new_data.append((v, i, fs, f0, device, *other))

    return new_data


def get_roi(signatures, i_on=0.05, i_running_min=0.1):
    ROI = []

    for v, i, fs, f0, app, *_ in tqdm(signatures):
        locs = get_running_interval(i, i_on, i_running_min)

        if locs is None:
            continue

        for on, off in locs:
            # Locations in periodic-domain
            a = on // i.shape[1]
            b = off // i.shape[1] + 1
            on = max(0, on - on // i.shape[1] * i.shape[1])
            off = min(off - on // i.shape[1] * i.shape[1],
                      i.shape[0] * i.shape[1] - 1)

            ROI.append((v[a:b], i[a:b], fs, f0, app, [[on, off]]))

    print(f'Submetered ROIs found: {len(ROI)}')

    return ROI


def assume_v(signatures, v_rms=120):
    signatures_new = []

    for _, i, fs, f0, app, *other in tqdm(signatures):
        t = np.linspace(0, len(i) * 2 * np.pi, np.prod(i.shape))
        v = v_rms * np.sqrt(2) * np.sin(t).reshape(*i.shape)
        v = v.astype(np.float32)
        signatures_new.append((v, i, fs, f0, app, *other))

    return signatures_new


def drop_noisy(signatures, thresh=10):
    mask = get_low_power_mask(signatures, P_min=thresh)
    mask = ~np.asarray(mask, dtype=bool)
    signatures = filter_with_mask(signatures, mask)

    return signatures


def drop_correlated(signatures, thresh=1e-4):
    I = get_total_current(signatures)
    I = I.reshape(len(I), -1)

    unique = np.ones(len(I), dtype=bool)
    D = 1 - cosine_similarity(I)

    for i in range(len(I)):
        if unique[i]:
            duplicate_indices = np.where(D[i] < thresh)[0]
            unique[duplicate_indices] = False
            unique[i] = True

    signatures = [s for s, u in zip(signatures, unique) if u]

    return signatures
