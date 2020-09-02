This package estimates parametric variograms from Delft3D-based RESQML models. The repository is tightly linked with https://github.com/NorskRegnesentral/nrresqml 

The main function is
<pre>
vargrest.estimate_variogram_parameters(settings, output_directory)
</pre>

## <tt>estimate_variogram_parameters</tt>

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

- **net_to_gross** A floating point value in millimeters used to generate a custom indicator value based on median
grain size per cell. net_to_gross provides a threshold such that net_to_gross < diameter will be used as an
indicator for producing sand. 0.088 is a reasonable testing value.

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

- **vargrest-output-&lt;I&gt;-\_crop\_.png** Image indicating the crop box used for execution &lt;I&gt;.

- **vargrest-output-&lt;I&gt;-\_data\_.pkl** Python pickle file containing some of the preliminary computation
results, including the full empirical variogram. This is primarily for debugging.

- **vargrest-output-&lt;I&gt;-\_slices\_.html** An html file showing the parameter values that the variogram is
based on. Shown layer-by-layer.

- **vargrest-output-&lt;I&gt;-\_variogram\_slices\_.png** Slices of the empirical and parametric variograms along
the X, Y and Z axes. Gives an indication of the parametric fit beyond the quality factor provided by summary.json.
Be aware that the slices are along the coordinate axes, not the major, minor and vertical axes of the variogram
ellipsoid.

- **vargrest-output-&lt;I&gt;-\_variograms\_2d\_.png** 2D slices of the empirical and parametric variograms along
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
one set of quality assessment files ("vargrest-output-*") are generated per configuration.

The main reason for doing multi-configuration execution is that results are gathered in a single folder and summary
files, making comparing results easier. Run time is not significantly better than doing single configuration
execution, as the empirical variogram must be re-computed for each configuration. Calculating the empirical
variogram is the time-consuming part. The exception is "family" which only relates to the parametric variogram, and
doing multi-configuration with multiple families of variograms will only do the empirical variogram estimation once.

### Advanced settings
The following settings are considered advanced in the vargrest package. It should not be necessary to adjust these
settings, but it can be useful to be aware of them:

- **family** Name of the parametric variogram family to be estimated. Valid names are spherical, exponential,
gaussian and general\_exponential. general\_exponential is always estimated with a power of 1.5, even though the
power could in principle be estimated as well. Default is to estimate for all families.

- **nugget** Boolean value (true/false) whether to estimate a nugget effect. Be aware that this might affect the
stability in finding a parametric fit to the empirical variogram. Default is False.

- **lagmax** Maximum estimation range in number of cells. Specified as a dictionary with keys "x", "y" and "z". Can
be used to reduce run time if the crop box is large, but the variogram range is expected to be small. Default is
half the size of the crop box.

- **sampling** Settings to control how sampling is done when estimating the empirical variogram. Specified as a
dictionary with a keyword "mode" and additional keywords depending the on the chosen mode. Three modes are
supported: dense, sparse and random. The main reason for choosing sparse or random is faster execution at the cost
of accuracy. However, accuracy has been more important than run time for version 1.0, and more effort has gone into
run-time optimization of the dense mode than the other two. Therefore, using other modes than dense may not yield
the expected run-time improvement. Default is to use dense sampling.

- **weighting** Dictionary with a single keyword, "sigma", with a floating point value. When the parametric
variogram is fitted to the empirical variogram, all data points are given equal weight. However, the proximal part
of the variogram is often the most interesting. As the range increases, more datapoints become available for
fitting, which increases the emphasis on the distant part of the variogram, instead of the proximal part. To
accommodate for this, a gaussian kernel weight can be applied to reduce the weight of the distant points in the
variogram estimation. The sigma keyword specifies the range of the gaussian kernel in number of cells. Default is
sigma = 10.0.

- **resample_dz** floating point value describing the resampling density in meters. Resampling refers to the
pre-processing step which maps the Delft3D-based grid onto a lattice grid. Default is 0.25.

- **full_qc** boolean flag to include full set of quality control data. Default is False.
