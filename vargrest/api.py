import itertools
import json
import os
import pathlib
from typing import Dict, Union, Optional, List, Any

from nrresqml.resqml import ResQml
from vargrest.variogramestimation.parametricvariogram import VariogramType
from vargrest.variogramresults import summary
from vargrest.variogramestimation.variogramestimation import VariogramEstimator, NonparametricVariogramEstimate
from vargrest.auxiliary.box import Box
from vargrest.variogramdata.variogramdata import VariogramDataInterface


def estimate_variogram_parameters(settings: Union[str, Dict], output_directory: str):
    if isinstance(settings, str):
        # Read settings from settings file
        settings = json.load(open(settings))
    data_file = settings['data_file']
    family = settings.get('family', None)
    nugget = settings.get('nugget', False)
    archel = settings.get('archel', None)
    cropbox = settings.get('cropbox', None)
    lagmax = settings.get('lagmax', None)
    indicator = settings.get('indicator', None)
    net_to_gross = settings.get('net_to_gross', None)
    sampling = settings.get('sampling', {'mode': 'dense', 'sub_sampling': None})
    weighting = settings.get('weighting', {'sigma': 10.0})
    resample_dz = settings.get('resample_dz', 0.25)
    attribute_name = settings.get('attribute_name', 'Porosity')

    # Make sure output directory exists, and if not, make sure that it can be created
    os.makedirs(output_directory, exist_ok=True)

    # Setup all estimation cases as lists
    # Set the box used to crop the variogram estimator's data
    if cropbox is None:
        boxes = [None]
    elif isinstance(cropbox, dict):
        boxes = [Box(cropbox['x_0'], cropbox['y_0'], cropbox['x_1'], cropbox['y_1'])]
    else:
        assert isinstance(cropbox, list)
        boxes = [Box(c['x_0'], c['y_0'], c['x_1'], c['y_1']) for c in cropbox]

    if family is None:
        families = [v for v in VariogramType]
    elif isinstance(family, list):
        families = [VariogramType(f) for f in family]
    else:
        families = [VariogramType(family)]

    if isinstance(archel, list):
        archels = archel
    else:
        archels = [archel]

    if indicator is None:
        indicators = []
    elif isinstance(indicator, list):
        indicators = indicator
    else:
        indicators = [indicator]
    if net_to_gross is not None:
        # Experimental functionality. API-control of custom indicators may change in the future
        assert isinstance(net_to_gross, (int, float))
        indicators.append(f'diameter<{net_to_gross}')

    if attribute_name is None:
        attribute_names = []
    elif isinstance(attribute_name, list):
        attribute_names = attribute_name
    else:
        attribute_names = [attribute_name]

    # Special iterator for attribute/indicator combination
    att_ind = [
        (_a, _i) for _a, _i in zip(
            attribute_names + [None] * len(indicators),
            [None] * len(attribute_names) + indicators
        )
    ]

    results = []
    # Read ResQml model file
    data_path = pathlib.Path(data_file)
    rq = ResQml.read_zipped(data_path)
    for _box, (_atr, _ind) in itertools.product(boxes, att_ind):
        # Read data from data file and create a variogram estimator
        rd = VariogramDataInterface.create_from_resqml(rq, _box, _atr, _ind)
        for _arc in archels:
            kwargs = {}
            if _arc is not None:
                kwargs['archels'] = [_arc]
            ve = VariogramEstimator(rd, dz=resample_dz, **kwargs)

            # Estimate empirical
            ne = _estimate_empirical(ve, lagmax, sampling)

            for _fam in families:
                # Get parametric variogram estimate
                sigma_wt = weighting['sigma']
                pe = ve.estimate_parametric_variogram_xyz(ne, family=_fam, nugget=nugget, sigma_wt=sigma_wt)

                # Conclude estimation and dump results
                i = len(results)
                summary.conclude(rd, ve, pe, ne, output_directory, f'vargrest_output-{i}-')
                md = {
                    summary.SummaryDataType.Identifier: i,
                    summary.SummaryDataType.Family: _fam.value,
                    summary.SummaryDataType.ArchelFilter: _arc,
                    summary.SummaryDataType.Indicator: _ind,
                    summary.SummaryDataType.Attribute: _atr if _ind is None else None,  # To avoid confusion
                    summary.SummaryDataType.Box: str(_box),
                }
                res = summary.summarize(pe, md)
                results.append(res)

    # Dump summary as csv
    summary.dump_summaries_to_csv(results, os.path.join(output_directory, 'summary.csv'))
    summary.dump_summaries_to_json(results, os.path.join(output_directory, 'summary.json'))


def _estimate_empirical(ve: VariogramEstimator, lagmax: Dict[str, int], sampling: Dict[str, Any]
                        ) -> NonparametricVariogramEstimate:
    # Determine extent (in lag distances) of empirical variogram estimate
    n_x, n_y, n_z = ve.data().shape
    if lagmax is None:
        l_x = int(0.5 * n_x)
        l_y = int(0.5 * n_y)
        l_z = int(0.5 * n_z)
    else:
        l_x = lagmax["x"]
        l_y = lagmax["y"]
        l_z = lagmax["z"]

    # Clamp lagmax to the grid (a longer lag has no effect)
    l_x = min(l_x, n_x)
    l_y = min(l_y, n_y)
    l_z = min(l_z, n_z)

    # Compute empirical variogram estimate
    samplingmode = sampling['mode']
    if samplingmode == "sparse":
        stride_x = sampling['stride_x']
        stride_y = sampling['stride_y']
        stride_z = sampling['stride_z']
        samplingstride = (stride_x, stride_y, stride_z)
        return ve.make_variogram_map_xyz(sampling="sparse", stride=samplingstride,
                                         lag_x=l_x, lag_y=l_y, lag_z=l_z)
    elif samplingmode == "random":
        samplingfactor = sampling['sampling_factor']
        maxsamples = sampling['max_samples']
        return ve.make_variogram_map_xyz(sampling="random", sampling_factor=samplingfactor, max_samples=maxsamples,
                                         lag_x=l_x, lag_y=l_y, lag_z=l_z)
    elif samplingmode == "dense":
        sub_sampling = sampling['sub_sampling']
        return ve.make_variogram_map_xyz(sampling="dense", sub_sampling=sub_sampling, lag_x=l_x, lag_y=l_y, lag_z=l_z)
    else:
        ValueError("Invalid sampling mode: {}. Must be dense, sparse or random.".format(samplingmode))
