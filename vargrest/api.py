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
    """
    Estimate variogram parameters based on the provided parameters

    ### settings
    File path to a json file or a dictionary containing estimation settings. All settings are optional except
    data_file. In addition to these settings, advanced settings are described below.

    - **data_file** File path to a ResQml model (.epc file)

    - **cropbox** Dictionary describing the extent of the model to use for estimation. Specified by providing keys x_0,
    x_1, y_0 and y_1 with float values. Delft3D models are typically starting at x=0, y=0.

    - **archel** An integer providing an architectural element to filter on before doing estimation. Values from other
    architectural elements will not contribute to the empirical variogram.

    - **indicator** An integer providing an architectural element to do indicator kriging on.

    - **net_to_gross** A floating point value in meters used to generate a custom indicator value based on median grain
    size per cell. Net_to_gross provides a threshold such that diameter < net_to_gross will be used as an indicator for
    producing sand. 0.088 is a reasonable testing value.

    - **attribute_name** Name of the parameter to do variogram estimation. Should typically be porosity or permeability.
    Default is to estimate porosity based on variables d50_per_sedclass and SedX_volfrac.

    ### output_directory
    Directory to which output is written. The following files are written (relative to the provided directory):

    - **settings.json** The settings used generate the output

    - **summary.json** List of results with each entry containing variogram range, azimuth, parametric fit quality, etc,
    as well as parameters identifying which settings were used. One list entry per execution (see multi-configuration
    below).

    - **summary.csv**: Similar to summary.json, but on a text-based table format. Tecnically not CSV as space is used as
    separator, not comma, and each column has a fixed width.

    - **vargrest-output-&lt;I&gt;-\\_crop\\_.png** Image indicating the crop box used for execution &lt;I&gt;.

    - **vargrest-output-&lt;I&gt;-\\_data\\_.pkl** Python pickle file containing some of the preliminary computation
    results, including the full empirical variogram. This is primarily for debugging.

    - **vargrest-output-&lt;I&gt;-\\_slices\\_.html** An html file showing the parameter values that the variogram is
    based on. Shown layer-by-layer.

    - **vargrest-output-&lt;I&gt;-\\_variogram\\_slices\\_.png** Slices of the empirical and parametric variograms along
    the X, Y and Z axes. Gives an indication of the parametric fit beyond the quality factor provided by summary.json.
    Be aware that the slices are along the coordinate axes, not the major, minor and vertical axes of the variogram
    ellipsoid.

    - **vargrest-output-&lt;I&gt;-\\_variograms\\_2d\\_.png** 2D slices of the empirical and parametric variograms along
    the coordinate axes. Anisotropic variograms that does not align with the coordinate axes, cannot be evaluated
    properly in the figure above. A 2D slice can give a better indication of the parametric fit, as it also visualizes
    the azimuth orientation.

    ### Multi-configuration execution
    Some of the keywords can be provided as lists of values, as well as single values, and invoke multi-configuration
    execution. The keywords that can be specified as lists are:
    - Cropbox
    - Archel
    - Indicator
    - Attribute_name
    - Family

    If one or more of these keywords are specified as lists, variogram estimates will be generated for all combinations
    of input values. The output files summary.json and summary.csv will contain one entry per configuration. Moreover,
    one set of quality assessment files (“vargrest-output-*”) are generated per configuration.

    The main reason for doing multi-configuration execution is that results are gathered in a single folder and summary
    files, making comparing results easier. Run time is not significantly better than doing single configuration
    execution, as the empirical variogram must be re-computed for each configuration. Calculating the empirical
    variogram is the time-consuming part. The exception is “family” which only relates to the parametric variogram, and
    doing multi-configuration with multiple families of variograms will only do the empirical variogram estimation once.

    ### Advanced settings
    The following settings are considered advanced in the vargrest package. It should not be necessary to adjust these
    settings, but it can be useful to be aware of them:

    - **family** Name of the parametric variogram family to be estimated. Valid names are spherical, exponential,
    gaussian and general\\_exponential. general\\_exponential is always estimated with a power of 1.5, even though the
    power could in principle be estimated as well. Default is to estimate for all families.

    - **nugget** Boolean value (true/false) whether to estimate a nugget effect. Be aware that this might affect the
    stability in finding a parametric fit to the empirical variogram. Default is False.

    - **lagmax** Maximum estimation range in number of cells. Specified as a dictionary with keys “x”, “y” and “z”. Can
    be used to reduce run time if the crop box is large, but the variogram range is expected to be small. Default is
    half the size of the crop box.

    - **sampling** Settings to control how sampling is done when estimating the empirical variogram. Specified as a
    dictionary with a keyword “mode” and additional keywords depending the on the chosen mode. Three modes are
    supported: dense, sparse and random. The main reason for choosing sparse or random is faster execution at the cost
    of accuracy. However, accuracy has been more important than run time for version 1.0, and more effort has gone into
    run-time optimization of the dense mode than the other two. Therefore, using other modes than dense may not yield
    the expected run-time improvement. Default is to use dense sampling.

    - **weighting** Dictionary with a single keyword, “sigma”, with a floating point value. When the parametric
    variogram is fitted to the empirical variogram, all data points are given equal weight. However, the proximal part
    of the variogram is often the most interesting. As the range increases, more datapoints become available for
    fitting, which increases the emphasis on the distant part of the variogram, instead of the proximal part. To
    accommodate for this, a gaussian kernel weight can be applied to reduce the weight of the distant points in the
    variogram estimation. The sigma keyword specifies the range of the gaussian kernel in number of cells. Default is
    sigma = 10.0.

    - **resample_dz** floating point value describing the resampling density in meters. Resampling refers to the
    pre-processing step which maps the Delft3D-based grid onto a lattice grid. Default is 0.25.

    - **full_qc** boolean flag to include full set of quality control data. Default is False.
    """
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
    full_qc = settings.get('full_qc', False)

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
                summary.conclude(rd, ve, pe, ne, output_directory, f'vargrest_output-{i}-', full_qc)
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
