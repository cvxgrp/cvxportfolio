Data cleaning
===================

.. automodule:: examples.data_cleaning

.. literalinclude:: ../../examples/data_cleaning.py
   :language: python
   :start-after: if __name__ ==
   :end-before: # we use this to save the plots
   :dedent:

This is the output printed to screen when executing this script. You can see
many diagnostics of the data cleaning. Scroll down to the pictures to see
what is actually going on.

.. include:: ../_static/data_cleaning_output.txt
   :literal:

And these are the figure that are plotted. For each of the test stocks, we
see the original Yahoo Finance data, which has many issues, and the cleaned
data that is produced, and used, by Cvxportfolio.

Data cleaning process for stock ``'SMT.L'``:

.. figure:: ../_static/SMT.L_data_cleaning.png
   :scale: 100 %
   :alt: Data cleaning process for stock ``'SMT.L'``

Data cleaning process for stock ``'NVR'``:

.. figure:: ../_static/NVR_data_cleaning.png
   :scale: 100 %
   :alt: Data cleaning process for stock 'NVR'

Data cleaning process for stock ``'HUBB'``:

.. figure:: ../_static/HUBB_data_cleaning.png
   :scale: 100 %
   :alt: Data cleaning process for stock 'HUBB'

Data cleaning process for stock ``'NWG.L'``:

.. figure:: ../_static/NWG.L_data_cleaning.png
   :scale: 100 %
   :alt: Data cleaning process for stock 'NWG.L'

Data cleaning process for stock ``'BA.L'``:

.. figure:: ../_static/BA.L_data_cleaning.png
   :scale: 100 %
   :alt: Data cleaning process for stock 'BA.L'
