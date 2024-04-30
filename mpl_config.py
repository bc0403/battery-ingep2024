import matplotlib as mpl
import matplotlib.font_manager as font_manager

# Set the backend if necessary
mpl.use('pdf')  # Uncomment this if all scripts should use the PDF backend

# Configure default settings for font, size, etc.
mpl.rcParams['pdf.fonttype'] = 42  # use Type 42 (a.k.a. TrueType) fonts
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']  # Assuming Helvetica is available
mpl.rcParams['font.family'] = 'sans-serif'
fonts = [f.name for f in font_manager.fontManager.ttflist]
if 'Helvetica' in fonts:
    print("Helvetica is available, used as the font of figure.")
elif 'Arial' in fonts:
    print("Helvetica is not available, use Arial as a substitute.")
elif 'sans-serif' in fonts:
    print("Both Helvetica and Arial are not available, use sans-serif instead.")
else:
    print('Please install sans-serif font family in your computer.')

# font size; Nature 7, IEEE 8
mpl.rcParams['font.size'] = 9  # 12  "approx. 9 of Helvetica is similar as 11 of Time New Roman"
mpl.rcParams['axes.labelsize'] = 9  # 14
mpl.rcParams['xtick.labelsize'] = 9  # 12
mpl.rcParams['ytick.labelsize'] = 9  # 12
mpl.rcParams['legend.fontsize'] = 9  # 10
# mpl.rcParams['figure.figsize'] = [3.487, 3.487/1.618]  # Default figure size
