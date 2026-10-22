"""Interactive dashboard for exploring reports and youth asthma across Chicago ZIP Code Zones"""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import shapefile
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

EXCESS_RISK_COLORS = dict(zip(
    ['< 0.25', '0.25 - 0.50', '0.50 - 1.00', '1.00 - 2.00',
     '2.00 - 4.00', '> 4.00', 'Undefined'],
    ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b', '#454545']))


def calculate_excess_risk(frame, events):
    """Compare observed events with counts expected from the pooled ZIP rate."""
    result = frame[['zip', events, 'Pop2020']].copy()
    counts = pd.to_numeric(result[events], errors='coerce')
    population = pd.to_numeric(result['Pop2020'], errors='coerce')
    valid = np.isfinite(counts) & counts.ge(0) & np.isfinite(population) & population.gt(0)
    result['Expected'] = np.nan
    result['Excess risk'] = np.nan
    result['Category'] = 'Undefined'
    if valid.any() and counts[valid].sum() > 0:
        reference_rate = counts[valid].sum() / population[valid].sum()
        result.loc[valid, 'Expected'] = population[valid] * reference_rate
        ratio = counts[valid] / result.loc[valid, 'Expected']
        result.loc[valid, 'Excess risk'] = ratio
        result.loc[valid, 'Category'] = np.select(
            [ratio < .25, ratio < .5, ratio < 1, ratio <= 2, ratio <= 4],
            list(EXCESS_RISK_COLORS)[:5], default='> 4.00')
    return result


def excess_risk_figure(mapped, geo, events):
    fig = px.choropleth_map(
        mapped, geojson=geo, locations='zip', color='Category',
        color_discrete_map=EXCESS_RISK_COLORS,
        category_orders={'Category': list(EXCESS_RISK_COLORS)},
        hover_name='zip',
        hover_data={'zip': False, events: ':,.0f', 'Pop2020': ':,.0f',
                    'Expected': ':.2f', 'Excess risk': ':.3f'},
        labels={events: 'Observed count', 'Pop2020': 'Population (2020)'},
        map_style='white-bg', center=CENTER, zoom=8.7,
    )
    fig.update_traces(marker_line_color='#9caaa9', marker_line_width=.8, showlegend=False)
    for rank, (category, color) in enumerate(EXCESS_RISK_COLORS.items()):
        count = int(mapped['Category'].eq(category).sum())
        fig.add_trace(go.Scattermap(
            lat=[None], lon=[None], mode='markers', marker=dict(size=12, color=color),
            name=f'{category} ({count})', showlegend=True, legendrank=rank, hoverinfo='skip'))
    fig.update_layout(
        font=dict(family='Inter, Arial, sans-serif', size=14), height=620,
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='white',
        map=dict(domain=dict(y=[.36, 1])),
        legend=dict(title='', orientation='v', x=0, y=.32,
                    xanchor='left', yanchor='top', itemclick=False, itemdoubleclick=False))
    return fig

def six_quantiles(values):
    """Return nullable Q1–Q6 codes and five cut points, retaining tied values.

    GeoDa uses Hazen percentiles, strict lower-half boundaries and inclusive
    upper-half boundaries (GeoDa Explore/CatClassification.cpp, quantile case).
    Repeated cut points leave empty categories; codes are never renumbered.
    """
    clean = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
    groups = pd.Series(pd.NA, index=values.index, dtype='Int64')
    valid = clean.dropna()
    if valid.empty:
        return groups, np.full(5, np.nan)
    breaks = np.quantile(valid, np.arange(1, 6) / 6, method='hazen')
    groups.loc[valid.index] = np.select(
        [valid < breaks[0], valid < breaks[1], valid < breaks[2],
         valid <= breaks[3], valid <= breaks[4]], [1, 2, 3, 4, 5], default=6)
    return groups, breaks


def classify_colocation(frame, housing, health):
    """Classify each indicator independently before comparing paired ZIPs."""
    result = frame[['zip', housing, health]].copy()
    for field in (housing, health):
        result[field] = pd.to_numeric(result[field], errors='coerce').replace(
            [np.inf, -np.inf], np.nan)
    hq, hb = six_quantiles(result[housing])
    eq, eb = six_quantiles(result[health])
    result['Housing quantile'] = hq.map(lambda q: f'Q{int(q)}' if pd.notna(q) else 'No data')
    result['Health quantile'] = eq.map(lambda q: f'Q{int(q)}' if pd.notna(q) else 'No data')
    paired = hq.notna() & eq.notna()
    same = (paired & hq.eq(eq)).fillna(False)
    result['Same quantile'] = 'Different quantiles'
    result.loc[same, 'Same quantile'] = result.loc[same, 'Housing quantile']
    result['Common top quantile'] = 'Not both in Q6'
    result.loc[(hq.eq(6) & eq.eq(6)).fillna(False), 'Common top quantile'] = 'Both in Q6'
    result.loc[~paired, ['Same quantile', 'Common top quantile']] = 'No data'
    return result, {housing: hb, health: eb}


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'sdoh_mockup_data'

st.set_page_config(page_title='Healthy Homes Chicago', layout='wide')

HOUSING = {'Lead inspection reporting': 'hdf24r', # lead inspection report rates per 10,000 population
           'Water in basement reporting': 'aaf24r', # rates per 10,000 population
           'Building violations': 'bba24r', # rates per 10,000 population
           'Rat reporting': 'sga24r'} # rates per 10,000 population
HEALTH = {'Youth Asthma ED Rate per 10,000 Youth': 'edvisr2024',
          'Youth Asthma ED Rate per 10,000 Residents': 'edvis24rc',
          'Youth Asthma ED Visit Count': 'edvis2024'}
TEAL = ['#e2f0ed', '#b9dbd4', '#8bc3b7', '#5aa595', '#307f70', '#155b50']
WARM = ['#f8e9dc', '#efcfaf', '#dfac80', '#ca8757', '#a96035', '#824223']
CENTER = {'lat': 41.84, 'lon': -87.72}

def display_title(text):
    title = text.title()
    for source, target in [('Ed', 'ED'), ('Zip', 'ZIP'), ('Lisa', 'LISA')]:
        title = title.replace(source, target)
    return title.replace(' Per ', ' per ')


def local_linear_fit(x, y, span=2 / 3):
    """Local linear fit with tricube distance weights, without robust reweighting."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    grid = np.linspace(x.min(), x.max(), 100)
    neighbors = min(len(x), max(3, int(np.ceil(span * len(x)))))
    fitted = []
    for target in grid:
        distance = np.abs(x - target)
        bandwidth = np.partition(distance, neighbors - 1)[neighbors - 1]
        if bandwidth == 0:
            fitted.append(y[distance == 0].mean())
            continue
        weights = (1 - np.minimum(distance / bandwidth, 1) ** 3) ** 3
        design = np.column_stack([np.ones(len(x)), x - target])
        root_weights = np.sqrt(weights)
        beta = np.linalg.lstsq(design * root_weights[:, None], y * root_weights, rcond=None)[0]
        fitted.append(beta[0])
    return grid, np.asarray(fitted)


def matrix_figure(sample, housing, outcome, outcome_label, method):
    fields = [*housing.values(), outcome]
    standard_deviation = sample[fields].std(ddof=1)
    standardized = (sample[fields] - sample[fields].mean()) / standard_deviation.replace(0, np.nan)
    standardized.loc[:, standard_deviation.eq(0)] = 0.0
    titles = [display_title(name.replace(' reporting', '')) for name in housing]
    fig = make_subplots(rows=1, cols=5,
                        horizontal_spacing=.04,
                        subplot_titles=titles + ['Youth Asthma ED Rates'])
    for column, field in enumerate(housing.values(), start=1):
        xx, yy = standardized[field].to_numpy(), standardized[outcome].to_numpy()
        fig.add_trace(go.Scatter(
            x=xx, y=yy, mode='markers', customdata=sample[['zip', field, outcome]].to_numpy(),
            marker=dict(size=6, color='#307f70', opacity=.7),
            hovertemplate='<b>ZIP %{customdata[0]}</b><br>Reporting Rate (Z-Score): %{x:.2f}'
                          '<br>Youth Asthma ED Rate (Z-Score): %{y:.2f}'
                          '<br>Reports / 10,000 Residents: %{customdata[1]:.2f}'
                          '<br>' + outcome_label + ': %{customdata[2]:.2f}<extra></extra>',
        ), row=1, col=column)
        if len(xx) >= 3 and np.unique(xx).size > 1:
            if method == 'Linear association':
                slope, intercept = np.polyfit(xx, yy, 1)
                grid = np.array([xx.min(), xx.max()])
                fitted = slope * grid + intercept
            elif method == 'Local regression':
                grid, fitted = local_linear_fit(xx, yy)
            else:
                grid = fitted = None
            if grid is not None:
                fig.add_trace(go.Scatter(x=grid, y=fitted, mode='lines',
                              line=dict(color='#a96035', width=2), hoverinfo='skip'), row=1, col=column)
        if len(xx) > 1 and np.std(xx) > 0 and np.std(yy) > 0:
            correlation = f'r = {np.corrcoef(xx, yy)[0, 1]:.3f}'
        else:
            correlation = 'r unavailable'
        fig.layout.annotations[column - 1].text += '<br><sup>' + correlation + '</sup>'
        fig.update_xaxes(title_text='Reporting Rate<br>(Z-Score)', showgrid=False, row=1, col=column)
        if column > 1:
            fig.update_yaxes(matches='y', showticklabels=False, row=1, col=column)
    counts, edges = np.histogram(standardized[outcome].to_numpy(), bins='sturges')
    fig.add_trace(go.Bar(
        x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges),
        customdata=np.column_stack([edges[:-1], edges[1:]]),
        marker=dict(color='#8bc3b7', line=dict(color='white', width=1)),
        hovertemplate='Youth Asthma ED Rate (Z-Score)<br>%{customdata[0]:.2f} – %{customdata[1]:.2f}'
                      '<br>Number of ZIP Zones: %{y}<extra></extra>',
    ), row=1, col=5)
    fig.update_xaxes(title_text='Youth Asthma ED Rate<br>(Z-Score)', showgrid=False, row=1, col=5)
    fig.update_yaxes(title_text='Number of ZIP Zones', rangemode='tozero', dtick=1,
                     row=1, col=5)
    fig.update_yaxes(title_text='Youth Asthma ED Rate<br>(Z-Score)', row=1, col=1)
    fig.update_yaxes(gridcolor='#edf0ef', zeroline=False)
    fig.update_xaxes(tickformat='.1f')
    for column in range(1, 5):
        fig.update_yaxes(tickformat='.1f', row=1, col=column)
    fig.update_yaxes(tickformat='d', row=1, col=5)
    fig.update_annotations(font_size=14)
    fig.update_layout(font=dict(family='Inter, Arial, sans-serif', size=14), height=310, showlegend=False, plot_bgcolor='white',
                      margin=dict(l=5,r=10,t=65,b=10))
    return fig


@st.cache_data
def load():
    reader = shapefile.Reader(str(DATA / 'chicago_zip'))
    frame = pd.DataFrame([record.as_dict() for record in reader.records()])
    frame['zip'] = frame['zip'].astype(str)
    # Use the supplied rates, including valid zeros, independently of missing
    # population or visit-count fields to match the outline's source data.
    geo = {'type': 'FeatureCollection', 
           'features': [
        {'type': 'Feature', 'id': str(record['zip']), 'properties': {}, 'geometry': shape.__geo_interface__}
        for record, shape in zip(reader.records(), reader.shapes())]}
    return frame, geo

@st.cache_data
def load_points(code):
    path = DATA / f'{code}_20_24cl.shp'
    if not path.exists():
        return None

    reader = shapefile.Reader(str(path))
    return pd.DataFrame([
        {
            'Year': record['year'],
            'ZIP code': str(record['ZIP_CODE']),
            'lon': record['LONGITUDE'],
            'lat': record['LATITUDE'],
        }
        for record in reader.records()
    ])


#basemap
def zip_base(geo, zip_codes):
    return go.Choroplethmap(
        geojson=geo, locations=zip_codes, z=[0] * len(zip_codes),
        colorscale=[[0, '#eef1ef'], [1, '#eef1ef']], showscale=False,
        marker_line_color='white', marker_line_width=.8,
        hovertemplate='ZIP %{location}<extra></extra>',
    )

#quantile map
def draw_map(frame, geo, field, title, unit, palette, mode):
    mode = mode.casefold()
    values = pd.to_numeric(frame[field], errors='coerce').replace([np.inf, -np.inf], np.nan)
    legend_labels = {}
    if mode == 'quantile map':
        ranks, _ = six_quantiles(values)
        groups = ranks.map(lambda v: f'Q{int(v)}' if pd.notna(v) else 'No data')
        colors = {f'Q{i+1}': color for i, color in enumerate(palette)}
        for category in colors:
            observed = values[groups.eq(category)]
            legend_labels[category] = (
                f'{category}: [{observed.min():.3f} : {observed.max():.3f}]'
                if not observed.empty else f'{category}: No observations')
    # box map
    elif mode == 'outlier map':
        q1, median, q3 = values.quantile([.25, .5, .75])
        iqr = q3-q1
        groups = np.select(
            [values.isna(), values < q1-1.5*iqr, values < q1,
             values < median, values <= q3, values <= q3+1.5*iqr],
            ['No data', 'Lower outlier', '< 25%', 
             '25% - 50%', '50% - 75%', '> 75%'], default='Upper outlier')
        colors = dict(zip(['Lower outlier', '< 25%', '25% - 50%', '50% - 75%', '> 75%', 'Upper outlier'],
                         ['#2166ac','#67a9cf','#d1e5f0','#fddbc7','#ef8a62','#b2182b']))
        edges = [-np.inf, q1 - 1.5 * iqr, q1, median, q3, q3 + 1.5 * iqr, np.inf]
        for category, lower, upper in zip(colors, edges, edges[1:]):
            legend_labels[category] = (
                f'[{lower:.3f} : {upper:.3f}]'
                if values.notna().any() else 'No valid data')
    else:
        raise ValueError(f'Unsupported map type: {mode}')
    colors['No data'] = '#e3e7e5'
    mapped = frame.assign(**{field: values}, Category=groups)
    fig = px.choropleth_map(
        mapped, geojson=geo, locations='zip', color='Category',
        color_discrete_map=colors, category_orders={'Category':list(colors)},
        hover_name='zip', hover_data={'zip':False, field:':.3f'}, labels={field:unit},
        map_style='white-bg', center=CENTER, zoom=8.7,
    )
    fig.update_traces(marker_line_color='white', marker_line_width=.8)
    fig.update_layout(font=dict(family='Inter, Arial, sans-serif', size=14), height=430, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='white',
                      legend=dict(title='',orientation='h',y=-.02,font_size=14))
    # Keep every class visible, including empty box-map outlier classes.
    fig.update_traces(showlegend=False)
    for rank, (category, color) in enumerate(colors.items()):
        count = int(mapped['Category'].eq(category).sum())
        if category == 'No data':
            if count == 0:
                continue
            label = f'No data ({count})'
        elif mode == 'quantile map':
            label = f'{legend_labels[category]} ({count})'
        else:
            label = f'{category}: {legend_labels[category]} ({count})'
        fig.add_trace(go.Scattermap(
            lat=[None], lon=[None], mode='markers',
            marker=dict(size=12, color=color), name=label, showlegend=True,
            legendrank=rank, hoverinfo='skip',
        ))
    fig.update_layout(
        height=620, map=dict(domain=dict(y=[.36, 1])),
        legend=dict(orientation='v', x=0, y=.32, xanchor='left', yanchor='top',
                    font_size=14, itemclick=False, itemdoubleclick=False),
    )
    with st.container(border=True):
        st.markdown('**'+title+'**')
        st.caption(unit)
        st.plotly_chart(fig,width='stretch',key=field,config={'displayModeBar':False})


@st.cache_data
def local_moran_results(frame, field, weights_path, weights_text):
    """Align by GAL IDs, validate loaded neighbors, and return results by ZIP.

    GAL text is an explicit cache input so a changed weights file invalidates
    results. Canonical file order also makes results invariant to table sorting.
    """
    import pygeoda

    lines = weights_text.splitlines()
    header = lines[0].split()
    if len(header) != 4 or header[0] != '0':
        raise ValueError('Expected a GAL header with an explicit ID field.')
    count, id_field = int(header[1]), header[3]
    tokens = iter(' '.join(lines[1:]).split())
    neighbors = {}
    try:
        for _ in range(count):
            identity, degree = next(tokens), int(next(tokens))
            if identity in neighbors or degree < 0:
                raise ValueError('Duplicate ID or invalid neighbor count in GAL file.')
            neighbors[identity] = [next(tokens) for _ in range(degree)]
    except StopIteration as exc:
        raise ValueError('Incomplete GAL weights file.') from exc
    if next(tokens, None) is not None:
        raise ValueError('Unexpected extra records in GAL weights file.')
    if id_field not in frame or frame[id_field].isna().any():
        raise ValueError(f'Weights require a complete {id_field} column.')
    indexed = frame.assign(_weight_id=frame[id_field].astype(str)).set_index('_weight_id')
    ids = list(neighbors)
    if not indexed.index.is_unique or set(indexed.index) != set(ids):
        raise ValueError('Data IDs do not match GAL IDs one-to-one.')
    for identity, adjacent in neighbors.items():
        if len(set(adjacent)) != len(adjacent) or identity in adjacent or not set(adjacent) <= set(ids):
            raise ValueError('Invalid neighbor IDs in GAL weights.')
        if any(identity not in neighbors[neighbor] for neighbor in adjacent):
            raise ValueError('Queen contiguity weights must be symmetric.')
    ordered = indexed.loc[ids]
    weights = pygeoda.read_gal(str(weights_path), ids)
    if weights.num_obs != len(ids):
        raise ValueError('Loaded weights have an unexpected observation count.')
    for i, identity in enumerate(ids):
        loaded = {ids[j] for j in weights.get_neighbors(i)}
        if loaded != set(neighbors[identity]):
            raise ValueError('Loaded weights do not match the GAL neighbor IDs.')
    values = pd.to_numeric(ordered[field], errors='coerce').replace([np.inf, -np.inf], np.nan)
    result = ordered[['zip']].copy()
    result[field] = values
    result['Local Moran I'] = np.nan
    result['Permutation p-value'] = np.nan
    result['Cluster'] = 'Undefined'
    result['Neighbors'] = [len(neighbors[identity]) for identity in ids]
    if values.notna().sum() >= 3 and values.nunique() > 1:
        lisa = pygeoda.local_moran(
            weights, values.fillna(0).tolist(), undefs=values.isna().tolist(),
            permutations=999, significance_cutoff=.05, seed=123456789,
            cpu_threads=6, permutation_method='complete')
        # Read labels from the installed library: numeric code order differs
        # between some documentation examples and the actual library.
        labels = lisa.lisa_labels()
        result['Cluster'] = [labels[code] for code in lisa.lisa_clusters()]
        result['Local Moran I'] = lisa.lisa_values()
        result['Permutation p-value'] = lisa.lisa_pvalues()
    result.loc[values.isna(), 'Cluster'] = 'Undefined'
    result.loc[values.notna() & result['Neighbors'].eq(0), 'Cluster'] = 'Isolated'
    unavailable = result['Cluster'].isin(['Undefined', 'Isolated'])
    result.loc[unavailable, ['Local Moran I', 'Permutation p-value']] = np.nan
    return result.reset_index(drop=True)


def local_moran_figure(mapped, geo, field, label):
    colors = {'Not significant': '#fafbfa', 'High-High': '#ff0000', 'Low-Low': '#0000ff',
              'Low-High': '#a7adf9', 'High-Low': '#f4ada8',
              'Undefined': '#626b73', 'Isolated': '#999999'}
    fig = px.choropleth_map(
        mapped, geojson=geo, locations='zip', color='Cluster',
        color_discrete_map=colors, category_orders={'Cluster': list(colors)},
        hover_name='zip', hover_data={'zip': False, field: ':.2f', 'Local Moran I': ':.3f',
                                    'Permutation p-value': ':.3f', 'Neighbors': True},
        labels={field: label}, map_style='white-bg', center=CENTER, zoom=8.7)
    fig.update_traces(marker_line_color='#9ba9a4', marker_line_width=.8, showlegend=False)
    for rank, (category, color) in enumerate(colors.items()):
        count = int(mapped['Cluster'].eq(category).sum())
        if category in ('Undefined', 'Isolated') and count == 0:
            continue
        fig.add_trace(go.Scattermap(
            lat=[None], lon=[None], mode='markers', marker=dict(size=11, color=color),
            name=f'{category} ({count})', legendrank=rank, showlegend=True, hoverinfo='skip'))
    fig.update_layout(font=dict(family='Inter, Arial, sans-serif', size=14), 
        height=640, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='white',
        map=dict(domain=dict(y=[.30, 1])),
        legend=dict(title='', orientation='v', x=0, y=.28, xanchor='left', yanchor='top',
                    font_size=14, itemclick=False, itemdoubleclick=False))
    return fig


def colocation_figure(mapped, geo, category, colors, labels):
    fig = px.choropleth_map(
        mapped, geojson=geo, locations='zip', color=category,
        color_discrete_map=colors, category_orders={category: list(colors)},
        hover_name='zip', hover_data={'zip': False, **{field: ':.2f' for field in labels},
                                    'Housing quantile': True, 'Health quantile': True},
        labels=labels, map_style='white-bg', center=CENTER, zoom=8.7,
    )
    fig.update_traces(marker_line_color='#9ba9a4', marker_line_width=.8, showlegend=False)
    for rank, (name, color) in enumerate(colors.items()):
        count = int(mapped[category].eq(name).sum())
        if name == 'No data' and not count:
            continue
        fig.add_trace(go.Scattermap(
            lat=[None], lon=[None], mode='markers', marker=dict(size=11, color=color),
            name=f'{name} ({count})', showlegend=True, legendrank=rank, hoverinfo='skip'))
    fig.update_layout(font=dict(family='Inter, Arial, sans-serif', size=14), 
        height=640, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='white',
        map=dict(domain=dict(y=[.30, 1])),
        legend=dict(title='', orientation='v', x=0, y=.28, xanchor='left', yanchor='top',
                    font_size=14, itemclick=False, itemdoubleclick=False))
    return fig


@st.cache_data
def reference_image(number):
    with ZipFile(ROOT / 'outline_ideas.docx') as archive:
        return archive.read(f'word/media/image{number}.png')


def references(numbers, titles):
    # Mock-up panels use fixed images from the supplied document.
    for col, number, title in zip(st.columns(len(numbers)), numbers, titles):
        with col:
            with st.container(border=True):
                st.markdown('**'+title+'**')
                st.image(reference_image(number),width='stretch')


HOUSING_NAMES = dict(zip(HOUSING, ['Lead inspections', 'Water in basements', 'Building violations', 'Rat reports']))
HEALTH_NAMES = dict(zip(HEALTH, ['Asthma ED visits per 10,000 youth',
                               'Asthma ED visits per 10,000 residents', 'Asthma ED visit count']))
df, geo = load()
st.html('''<style>
    .stApp [data-testid="stMainBlockContainer"] {
        padding-top: 3rem;
    }
    .stApp [data-testid="stVerticalBlockBorderWrapper"] {
        border-width: 1.5px !important;
    }
    .stApp [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
        border-width: 1.5px;
    }
    .stApp hr { border-bottom-width: 1.5px; }
    .st-key-comparison_controls {
        background-color: #F7F9F8;
        border: 1px solid #E0E7E3 !important;
        border-radius: 12px;
        padding: 20px 24px;
        gap: 12px;
    }
    .st-key-comparison_controls [data-baseweb="select"] > div {
        background-color: #FFFFFF;
    }
    .comparison-controls-heading {
        margin: 0 0 4px;
        font-size: 18px;
        font-weight: 600;
        line-height: 1.4;
    }
    .comparison-controls-description {
        margin: 0;
        color: #66736D;
        font-size: 14px;
        line-height: 1.5;
    }
</style>''')
st.title('Healthy Homes', anchor=False)
st.write('Explore where healthy homes concerns (based on 311 reports) and youth asthma emergency department (ED) visits '
         'are concentrated across Chicago, and how their patterns overlap')
with st.expander('Where are healthy homes concerns located? (311 Point map)', expanded=True):
    point_housing = st.selectbox('Healthy homes concerns', list(HOUSING), key='point_housing', format_func=HOUSING_NAMES.get)
    points = load_points(HOUSING[point_housing][:3])
    fig = go.Figure(zip_base(geo, df['zip']))
    if points is not None and not points.empty:
        year_options = sorted(points['Year'].dropna().unique().tolist())
        zip_options = sorted(set(df['zip']) | set(points['ZIP code'].dropna()))
        year_column, area_column = st.columns(2)
        with year_column:
            selected_years = st.multiselect(
                'Years', year_options, default=year_options,
                key='point_years_' + HOUSING[point_housing],
            )
        with area_column:
            selected_zips = st.multiselect(
                'ZIP areas', zip_options, key='point_zip_areas',
                placeholder='All ZIP areas',
            )
        filtered_points = points.loc[points['Year'].isin(selected_years)]
        if selected_zips:
            filtered_points = filtered_points.loc[filtered_points['ZIP code'].isin(selected_zips)]
            fig.add_trace(go.Choroplethmap(
                geojson=geo, locations=selected_zips, z=[1] * len(selected_zips),
                colorscale=[[0, '#cfe5df'], [1, '#cfe5df']], showscale=False,
                marker_line_color='#307f70', marker_line_width=1.5,
                hovertemplate='Selected ZIP %{location}<extra></extra>',
            ))
        st.metric('Reports in selected years and ZIP areas', f'{len(filtered_points):,}')
        if filtered_points.empty:
            st.info('No reports match the selected years and ZIP areas.')
        fig.add_trace(go.Scattermap(lat=filtered_points['lat'],lon=filtered_points['lon'],mode='markers',
            marker=dict(size=5,color='#307f70',opacity=.65),customdata=filtered_points[['Year','ZIP code']],
            hovertemplate='Year: %{customdata[0]}<br>ZIP: %{customdata[1]}<extra></extra>'))
        with st.expander('Report counts by ZIP area'):
            counts = filtered_points.groupby('ZIP code').size()
            if selected_zips:
                counts = counts.reindex(selected_zips, fill_value=0)
            summary = counts.rename('Reports').rename_axis('ZIP code').reset_index()
            st.dataframe(summary, hide_index=True, width='stretch')
    else:
        st.info('Point data are currently unavailable for this report type.')
    fig.update_layout(font=dict(family='Inter, Arial, sans-serif', size=14), map=dict(style='white-bg',center=CENTER,zoom=8.9),height=420,
                      margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
    st.plotly_chart(fig,width='stretch',key='points',config={'displayModeBar':False})

st.space(8)
st.divider()
st.space(8)
with st.container(border=True, key='comparison_controls'):
    st.html('<div class="comparison-controls-heading">Choose measures to compare</div>'
            '<p class="comparison-controls-description">Applies to the interactive comparisons below</p>')
    left, right = st.columns(2)
    with left:
        hx = st.selectbox('Healthy homes concerns', list(HOUSING), key='housing', format_func=HOUSING_NAMES.get)
    with right:
        hy = st.selectbox('Youth asthma measure', list(HEALTH), key='health', format_func=HEALTH_NAMES.get)
st.space(16)
x = HOUSING[hx]
housing_label = HOUSING_NAMES[hx]
y = HEALTH[hy]
health_label = HEALTH_NAMES[hy]
with st.expander('How do areas compare?', expanded=True):
    map_view = st.segmented_control('Area comparison', ['Quantile maps', 'Box / Outlier maps', 'Excess Risk'],
        default='Quantile maps', required=True, key='area_map_view', label_visibility='collapsed')
    if map_view == 'Excess Risk':
        housing_events = {'hdf24r': 'hdf2024', 'aaf24r': 'aaf2024',
                          'bba24r': 'bba2024', 'sga24r': 'sga2024'}[x]
        for col, events, title in zip(st.columns(2), [housing_events, 'edvis2024'],
                                      [housing_label, 'Youth asthma ED visits']):
            with col, st.container(border=True):
                st.markdown('**' + title + '**')
                rate_unit = ('Reports per 10,000 residents' if events == housing_events
                             else 'Youth asthma ED visits per 10,000 total residents')
                st.caption(rate_unit + '  \nPopulation: 2020 total population')
                mapped = calculate_excess_risk(df, events)
                st.plotly_chart(excess_risk_figure(mapped, geo, events), width='stretch',
                                key='excess_' + events, config={'displayModeBar': False})
        st.markdown('**How to read**')
        st.write('Excess risk = observed / expected count')
        st.write('Observed is the recorded number of reports or ED visits in a ZIP; expected is '
                 'the ZIP\'s population multiplied by the overall rate '
                 '(total reports or visits / total population across ZIP zones with valid data)')
        st.write('To interpret the ratio, for example, 1 means the observed count matches the expected count, '
         '2 means twice the expected count, and 0.5 means half the expected count, undefined means missing or invalid data')
        st.write('(For Excess Risk maps, the left map uses the concern type selected under Healthy homes concerns; '
         'the right map always uses youth asthma ED visit counts and total population, '
         'regardless of the Youth asthma measure selection)')
    else:
        mode = 'quantile map' if map_view == 'Quantile maps' else 'outlier map'
        left, right = st.columns(2)
        with left: draw_map(df, geo, x, housing_label, 'Reports per 10,000 residents', WARM, mode)
        with right: draw_map(df, geo, y, 'Youth asthma', health_label, TEAL, mode)
        with st.container():
            st.markdown('**How to read**')
            if map_view == 'Quantile maps':
                st.write('Quantile maps divide ZIP zones into six groups based on their relative values, '
                         'from lowest (Q1) to highest (Q6). '
                         'Darker colors show higher values within each measure. '
                         'Legend ranges show the minimum and maximum values in each group '
                         'and numbers in parentheses show the number of ZIP zones')
            else:
                st.write('Box maps show which ZIP zones have unusually high or low values. '
                         'Legend ranges show class boundaries '
                         'and numbers in parentheses show the number of ZIP zones')

st.space(8)
with st.expander('Where do high values overlap? (Co-location maps)'):
    asthma_unit = {
        'edvisr2024': 'visits per 10,000 youth',
        'edvis24rc': 'visits per 10,000 residents',
        'edvis2024': 'total visits',
    }[y]
    mapped, breaks = classify_colocation(df, x, y)
    labels = {x: housing_label + ' per 10,000 residents', y: health_label}
    palettes = [
        {'Different quantiles': '#fafbfa', **dict(zip([f'Q{i}' for i in range(1,7)],
            ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c'])), 'No data':'#626b73'},
        {'Both in Q6':'#e31a1c','Not both in Q6':'#fafbfa','No data':'#626b73'}]
    for col, category, title, palette in zip(st.columns(2), ['Same quantile','Common top quantile'],
            ['Similar rankings (Quantile co-location)', 'Both among the highest (Common top quantile)'], palettes):
        with col, st.container(border=True):
            st.markdown('**' + title + '**')
            st.caption(f'{housing_label} (reports per 10,000 residents)  \n'
                       f'Youth asthma ED visits ({asthma_unit})')
            fig = colocation_figure(mapped, geo, category, palette, labels)
            st.plotly_chart(fig, width='stretch', key='colocation_' + category, config={'displayModeBar':False})
    with st.container():
        st.markdown('**How to read**')
        st.write('On the left, colored ZIP zones show where both measures fall in the same quantile group. '
                 'On the right, only ZIP zones where both measures fall in the highest group (Q6) are highlighted in red '
                 'and all others are light gray')
        st.write('These maps show geographic overlap, not necessarily spatial clustering or causation')
    for field in [x,y]:
        if mapped[field].nunique() < 2 or len(np.unique(breaks[field])) < 5:
            st.caption('Some groups are empty or have tied values. Read the map as relative rankings.')

st.space(8)
with st.expander('Explore relationships (Scatterplots and Regression)'):
    analysis = st.selectbox('Analysis', ['Relationship between measures', 'Compare housing indicators'], key='relationship_view')
    if analysis == 'Relationship between measures':
        valid=df.dropna(subset=[x,y])
        st.caption(f'n = {len(valid)} ZIP Code Zones')
        x_label = housing_label + ' per 10,000 residents'
        y_label = hy
        fig=px.scatter(valid,x=x,y=y,hover_name='zip',color_discrete_sequence=['#307f70'],
                       labels={x:x_label,y:y_label})
        fig.update_traces(marker=dict(size=8,opacity=.75))
        if len(valid)>=3 and valid[x].nunique()>1:
            slope,intercept=np.polyfit(valid[x],valid[y],1)
            xx=np.array([valid[x].min(),valid[x].max()])
            fig.add_trace(go.Scatter(x=xx,y=slope*xx+intercept,mode='lines',
                                    name='Linear trend',line=dict(color='#a96035',width=2)))
        fig.update_layout(font=dict(family='Inter, Arial, sans-serif', size=14), height=340,showlegend=False,plot_bgcolor='white',
                          margin=dict(l=0,r=0,t=10,b=10))
        # Leave room below zero so points at the origin remain visible.
        for axis in ['x', 'y']:
            plotted = np.concatenate([np.asarray(getattr(trace, axis), dtype=float) for trace in fig.data])
            finite = plotted[np.isfinite(plotted)]
            maximum = max(float(finite.max()) if finite.size else 0, 1)
            target_step = maximum / 5
            magnitude = 10 ** np.floor(np.log10(target_step))
            step = next(multiplier * magnitude for multiplier in [1, 2, 2.5, 5, 10]
                        if multiplier * magnitude >= target_step)
            upper = np.ceil(maximum * 1.05 / step) * step
            lower = min(float(finite.min()) if finite.size else 0, 0) - float(upper) * .05
            settings = dict(range=[lower, float(upper)], autorange=False, tickmode='linear',
                            tick0=0, dtick=float(step), showline=True, linecolor='#c7d0cb',
                            zeroline=False)
            if axis == 'x':
                fig.update_xaxes(showgrid=False, **settings)
            else:
                fig.update_yaxes(gridcolor='#edf0ef', **settings)
        fig.add_vline(x=0, line_dash='dash', line_color='#8a9691', line_width=1, layer='below')
        fig.add_hline(y=0, line_dash='dash', line_color='#8a9691', line_width=1, layer='below')
        st.plotly_chart(fig,width='stretch',key='relationship',config={'displayModeBar':False})
        st.markdown('**How to read**')
        st.write('Each point is a ZIP zone; the horizontal axis shows reports per 10,000 residents for the selected '
                 'report type, and the vertical axis shows the selected youth asthma measure')
        st.write('The chart shows whether ZIP zones with higher reporting rates also tend to have higher asthma ED visit values. ' 
                 'An upward trend suggests a positive association but does not imply causation')
    else:
        compare_y = 'edvisr2024'
        compare_health = 'Youth Asthma ED Rate per 10,000 Youth'
        compare_sample = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[*HOUSING.values(), compare_y])
        st.caption('This comparison always uses asthma ED visits per 10,000 youth and report rates per 10,000 residents for all four report types, regardless of the selections above')
        method = st.selectbox('Show', ['Scatterplot matrix', 'Linear association', 'Local regression'])
        st.caption('Scatterplot matrix shows the ZIP points alone; Linear association adds one straight trend line; Local regression adds a curve that follows nearby points')
        fig = matrix_figure(compare_sample, HOUSING, compare_y, compare_health, method)
        st.plotly_chart(fig, width='stretch', key='compare_' + method, config={'displayModeBar':False})
        st.markdown('**How to read**')
        st.write('Each point is a ZIP zone; a Z-score of 0 is average, positive is above average, '
                 'and negative is below average')
        st.write('A point with both x and y above 0 has higher report and youth asthma ED visit '
                 'rates than their respective averages')
        st.write('The last panel shows how youth asthma rates are distributed across ZIP zones')
        st.write('The two rates use different populations and are standardized separately, so their '
                 'raw values should not be compared directly')
        st.write('A trend here shows association across ZIP zones, not causation')

st.space(8)
with st.expander("Where do nearby areas share a pattern? (Local Moran's I)"):
    weights_path = DATA / 'chicago_zip_q.gal'
    try:
        weights_text = weights_path.read_text()
        results = [local_moran_results(df, field, str(weights_path), weights_text) for field in [x,y]]
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        st.error('The neighborhood analysis is currently unavailable.')
        with st.expander('Technical details'):
            st.write(str(exc))
    else:
        for col, field, label, result in zip(st.columns(2), [x,y],
                [housing_label + ' per 10,000 residents', health_label], results):
            with col, st.container(border=True):
                st.markdown('**' + label + '**')
                if result['Cluster'].eq('Undefined').all():
                    st.info('There is not enough variation or available data to identify clusters.')
                fig = local_moran_figure(result, geo, field, label)
                st.plotly_chart(fig, width='stretch', key='local_moran_' + field, config={'displayModeBar':False})
    with st.container():
        st.markdown('**How to read**')
        st.write('These maps show clusters of higher or lower values for the two selected measures')
        st.write('High–High means a ZIP has a value above the average and is surrounded by higher values. '
                 'Low–Low shows the reverse. High–Low and Low–High identify ZIP zones that differ from their neighbors. '
                 'High and low refer to the mean of each measure')
        st.write('Not significant means this test did not identify a clear spatial pattern. '
                 'It does not mean there are no housing or health concerns. Undefined means data needed for the calculation are missing')
        st.caption('Local Moran’s I (Queen contiguity, 999 permutations, significance level 0.05, uncorrected)')

st.space(8)
with st.expander('Where do the highest values cluster? (Quantile LISA)'):
    references([17, 18, 19], [
        'Youth asthma ED visits per 10,000 youth (edvisr2024)',
        'Lead inspection reports per 10,000 residents (hdf24r)',
        'Joint clusters: lead inspection reporting and youth asthma ED rates (hdf24r + edvisr2024)',
    ])
