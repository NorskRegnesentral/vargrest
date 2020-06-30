def progress(itr, *args, **kwargs):
    try:
        import tqdm
        return tqdm.tqdm(itr, *args, **kwargs)
    except ImportError:
        return itr