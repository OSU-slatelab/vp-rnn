from setuptools import setup, find_packages
setup(
    name='vp-rnn',

    version='0.0.1',
    description='Self-Attentive RNNs for Virtual Patient question classification',

    # Author details
    author='Adam Stiff',
    
    # Choose your license
    license='Apache 2.0',

    packages=find_packages(),
    
    python_requires='>=3',
    install_requires=['torch==1.4.0', 'scikit-learn', 'numpy'],

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.

    package_data={
        'vp-rnn': ['data/*'],
    },

)
