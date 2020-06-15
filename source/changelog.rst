.. _Changelog:

What's new
##########

.. contents:: Table of Contents
   :depth: 2

v0.1.1 (June 2020)
------------------

**New functions**

a. Add the **plotly** sub-module, a set of Plotly functions comprising :py:func:`systole.plotly.plot_raw`, :py:func:`systole.plotly.plot_subspaces`, :py:func:`systole.plotly.plot_ectopic`, :py:func:`systole.plotly.plot_shortLong`, :py:func:`systole.plotly.plot_frequency`, :py:func:`systole.plotly.plot_nonlinear`, :py:func:`systole.plotly.plot_timedomain`.
b. Add :py:func:`plotly.utils.simulate_rr()`, for random RR interval simulation with different kind of artefacts. Can also return peak vector.
c. The **correction** sub-module has been largely rewritten and now include :py:func:`systole.correction.correct_extra`, :py:func:`systole.correction.correct_missed`, :py:func:`systole.correction.interpolate_bads`, :py:func:`systole.correction.correct_rr`, :py:func:`systole.correction.correct_peaks`, :py:func:`systole.correction.correct_missed_peaks`, :py:func:`systole.correction.correct_extra_peaks`. These function can correct artefacts either using peaks addition/removal or by interpolation of the RR time series.

**Enhancements**

a. The **detection** sub-module has been improved. It is now about 10x faster and return more information. The main function has been renamed to :py:func:`systole.detection.rr_artefacts`.

**Bugfixes**

a. :py:func:`systole.correction.interpolate_clipping`: add exception in case of clipping artefacts at the edge of the signal segment. This can cause cash during recording. The default behavior is now to decrement the last/first item in case of threshold value. The threshold can be changed manually. This procedure can result in slightly inaccurate interpolation, using a longer recording should always be preferred when possible.
b. The PPG signal simulator used for testing can now run infinitely.

**Contributors**

* `Jan C. Brammer <jan.c.brammer@gmail.com>`_


v0.1.0 (January 2020)
---------------------

Initial release.

**Detection**

a. oxi_peaks()
b. hr_subspaces()
c. interpolate_clipping()
d. rr_outliers()


**HRV**

a. nnX()
b. pnX()
c. rmssd()
d. time_domain()
e. frequency_domain()
f. nonlinear()


**Plotting**

a. plot_hr()
b. plot_events()
c. plot_oximeter()
d. plot_subspaces()
e. plot_psd()
f. circular()
g. plot_circular()


**Recording**

a. Oximeter()


**Report**

a. report_oxi()


**Utils**

a. norm_triggers()
b. time_shift()
c. heart_rate()
d. to_angles()
e. to_epochs()

v0.0.1 (January 2020)
---------------------

Alpha release.
