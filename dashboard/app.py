"""Interactive dashboard for exploring housing reports and youth asthma across Chicago ZIP Code Zones"""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import shapefile
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'sdoh_mockup_data'

st.set_page_config(page_title='Healthy Homes Chicago', layout='wide')

HOUSING = {'Lead inspection reporting': 'hdf24r', # lead inspection report rates per 10,000 population
           'Water in basement reporting': 'aaf24r', # rates per 10,000 population
           'Building violation reporting': 'bba24r', # rates per 10,000 population
           'Rat reporting': 'sga24r'} # rates per 10,000 population
HEALTH = {'Asthma ED rate per 10,000 youth': 'edvisr2024',
          'Asthma ED rate per 10,000 residents': 'edvis24rc', 
          'Asthma ED visit count': 'edvis2024'}
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
                        subplot_titles=titles + ['Asthma ED Visit Rates'])
    for column, field in enumerate(housing.values(), start=1):
        xx, yy = standardized[field].to_numpy(), standardized[outcome].to_numpy()
        fig.add_trace(go.Scatter(
            x=xx, y=yy, mode='markers', customdata=sample[['zip', field, outcome]].to_numpy(),
            marker=dict(size=6, color='#307f70', opacity=.7),
            hovertemplate='<b>ZIP %{customdata[0]}</b><br>Reporting Rate (Z-Score): %{x:.2f}'
                          '<br>Asthma Rate (Z-Score): %{y:.2f}'
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
        hovertemplate='Asthma Rate (Z-Score)<br>%{customdata[0]:.2f} – %{customdata[1]:.2f}'
                      '<br>Number of ZIP Zones: %{y}<extra></extra>',
    ), row=1, col=5)
    fig.update_xaxes(title_text='Asthma ED Visit Rate<br>(Z-Score)', showgrid=False, row=1, col=5)
    fig.update_yaxes(title_text='Number of ZIP Zones', rangemode='tozero', dtick=1,
                     row=1, col=5)
    fig.update_yaxes(title_text='Asthma ED Visit Rate<br>(Z-Score)', row=1, col=1)
    fig.update_yaxes(gridcolor='#edf0ef', zeroline=False)
    fig.update_xaxes(tickformat='.1f')
    for column in range(1, 5):
        fig.update_yaxes(tickformat='.1f', row=1, col=column)
    fig.update_yaxes(tickformat='d', row=1, col=5)
    fig.update_annotations(font_size=13)
    fig.update_layout(height=310, showlegend=False, plot_bgcolor='white',
                      margin=dict(l=5,r=10,t=65,b=10))
    return fig


@st.cache_data
def load():
    reader = shapefile.Reader(str(DATA / 'chicago_zip'))
    frame = pd.DataFrame([record.as_dict() for record in reader.records()])
    frame['zip'] = frame['zip'].astype(str)
    frame.loc[frame.Pop2020.isna() | (frame.Pop2020 <= 0), [*HOUSING.values(), 'edvis24rc']] = np.nan
    frame.loc[frame.edvis2024.isna(), ['edvisr2024', 'edvis24rc']] = np.nan
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
    values = frame[field]
    if mode == 'quantile map':
        ranks = pd.qcut(values, 6, labels=False, duplicates='drop')
        groups = ranks.map(lambda v: f'Q{int(v)+1}' if pd.notna(v) else 'No data')
        colors = {f'Q{i+1}': color for i, color in enumerate(palette)}
    # box map
    elif mode == 'outlier map':
        q1, median, q3 = values.quantile([.25, .5, .75])
        iqr = q3-q1
        groups = np.select(
            [values.isna(), values < q1-1.5*iqr, values <= q1, 
             values <= median, values <= q3, values <= q3+1.5*iqr],
            ['No data', 'Lower outlier', '< 25%', 
             '25% - 50%', '50% - 75%', '> 75%'], default='Upper outlier')
        colors = dict(zip(['Lower outlier', '< 25%', '25% - 50%', '50% - 75%', '> 75%', 'Upper outlier'],
                         ['#2166ac','#67a9cf','#d1e5f0','#fddbc7','#ef8a62','#b2182b']))
    else:
        raise ValueError(f'Unsupported map type: {mode}')
    colors['No data'] = '#e3e7e5'
    mapped = frame.assign(Category=groups)
    fig = px.choropleth_map(
        mapped, geojson=geo, locations='zip', color='Category',
        color_discrete_map=colors, category_orders={'Category':list(colors)},
        hover_name='zip', hover_data={'zip':False, field:':.2f'}, labels={field:unit},
        map_style='white-bg', center=CENTER, zoom=8.7,
    )
    fig.update_traces(marker_line_color='white', marker_line_width=.8)
    fig.update_layout(height=430, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='white',
                      legend=dict(title='',orientation='h',y=-.02,font_size=11))
    if mode == 'quantile map':
        counts = mapped['Category'].value_counts()
        for trace in fig.data:
            trace.name = f'{trace.name} ({int(counts.get(trace.name, 0))})'
    if mode == 'outlier map':
        # Include all six classes, even when a class contains no ZIP zones
        fig.update_traces(showlegend=False)
        for rank, (category, color) in enumerate(colors.items()):
            count = int((mapped['Category'] == category).sum())
            if category == 'No data' and count == 0:
                continue
            fig.add_trace(go.Scattermap(
                lat=[None], lon=[None], mode='markers',
                marker=dict(size=12, color=color),
                name=f'{category} ({count})', showlegend=True,
                legendrank=rank, hoverinfo='skip',
            ))
        fig.update_layout(
            height=620, map=dict(domain=dict(y=[.36, 1])),
            legend=dict(orientation='v',
                        x=0, y=.32, xanchor='left', yanchor='top',
                        font_size=12, itemclick=False, itemdoubleclick=False),
        )
    with st.container(border=True):
        st.markdown('**'+display_title(title)+'**')
        st.caption(unit)
        st.plotly_chart(fig,width='stretch',key=field,config={'displayModeBar':False})


@st.cache_data
def reference_image(number):
    with ZipFile(ROOT / 'outline_ideas.docx') as archive:
        return archive.read(f'word/media/image{number}.png')


def references(numbers, titles):
    # Mock-up panels use fixed images from the supplied document.
    for col, number, title in zip(st.columns(len(numbers)), numbers, titles):
        with col:
            with st.container(border=True):
                st.markdown('**'+display_title(title)+'**')
                st.image(reference_image(number),width='stretch')

st.title('Healthy Homes')
st.write('Explore housing reports and youth asthma across Chicago ZIP Code Zones')
df, geo = load()

with st.container(border=True, key='selected_indicators'):
    st.subheader('Select Indicators', anchor=False)
    housing_control, health_control = st.columns(2)
    with housing_control:
        hx = st.selectbox('311 Call Type', list(HOUSING), key='housing', format_func=display_title)
    with health_control:
        hy = st.selectbox('Asthma Measure', list(HEALTH), key='health', format_func=display_title,
                          help='ED = Emergency Department')
    x, y = HOUSING[hx], HEALTH[hy]
    unit = {'edvisr2024':'Asthma ED Visits per 10,000 Youth',
            'edvis24rc':'Asthma ED Visits per 10,000 Residents', 'edvis2024':'Number of Asthma ED Visits'}[y]
    with st.container(key='maps_section'):
        st.header('Maps', anchor=False)
        mode = st.segmented_control('Map view',
            ['Quantile Map','Outlier Map','311 Point Map','Excess Risk'],
            default='Quantile Map',required=True,label_visibility='collapsed')

        if mode.casefold() in ['quantile map','outlier map']:
            left,right=st.columns(2)
            with left: draw_map(df,geo,x,hx,'Reports per 10,000 residents',WARM,mode)
            with right: draw_map(df,geo,y,'Youth asthma',unit,TEAL,mode)
        elif mode.casefold()=='311 point map':
            points=load_points(x[:3])
            with st.container(border=True):
                st.markdown('**' + display_title(hx.replace(' reporting', ' reports')) + ' (2020–2024)**')
                fig=go.Figure(zip_base(geo,df['zip']))
                if points is not None:
                    fig.add_trace(go.Scattermap(lat=points['lat'],lon=points['lon'],mode='markers',
                        marker=dict(size=5,color='#307f70',opacity=.65),customdata=points[['Year','ZIP code']],
                        hovertemplate='Year: %{customdata[0]}<br>ZIP: %{customdata[1]}<extra></extra>'))
                    st.caption(f'{len(points):,} reports')
                else:
                    st.caption('N/A')
                fig.update_layout(map=dict(style='white-bg',center=CENTER,zoom=8.7),height=430,
                                  margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
                st.plotly_chart(fig,width='stretch',key='points',config={'displayModeBar':False})
        elif mode.casefold() == 'excess risk':
            st.caption('Based on 2020 Population')
            references([6,7], ['Lead Inspection Reports', 'Youth Asthma ED Visits'])

    st.divider()
    st.header('Relationship', anchor=False)
    st.subheader('Youth Asthma ED Visits vs. ' + display_title(hx.replace(' reporting', ' reports')), anchor=False)
    valid=df.dropna(subset=[x,y])
    st.caption(f'n = {len(valid)} ZIP Code Zones')
    x_label = display_title(hx.replace(' reporting', ' reports')) + ' per 10,000 Residents'
    y_label = {
        'edvisr2024': 'Youth Asthma ED Visits per 10,000 Youth',
        'edvis24rc': 'Youth Asthma ED Visits per 10,000 Residents',
        'edvis2024': 'Youth Asthma ED Visits (Count)',
    }[y]
    fig=px.scatter(valid,x=x,y=y,hover_name='zip',color_discrete_sequence=['#307f70'],
                   labels={x:x_label,y:y_label})
    fig.update_traces(marker=dict(size=8,opacity=.75))
    if len(valid)>=3 and valid[x].nunique()>1:
        slope,intercept=np.polyfit(valid[x],valid[y],1)
        xx=np.array([valid[x].min(),valid[x].max()])
        fig.add_trace(go.Scatter(x=xx,y=slope*xx+intercept,mode='lines',
                                name='Linear trend',line=dict(color='#a96035',width=2)))
    fig.update_layout(height=340,showlegend=False,plot_bgcolor='white',
                      margin=dict(l=0,r=0,t=10,b=10))
    # Set complete numeric ranges so the browser cannot add space below zero
    for axis in ['x', 'y']:
        plotted = np.concatenate([np.asarray(getattr(trace, axis), dtype=float) for trace in fig.data])
        finite = plotted[np.isfinite(plotted)]
        maximum = max(float(finite.max()) if finite.size else 0, 1)
        target_step = maximum / 5
        magnitude = 10 ** np.floor(np.log10(target_step))
        step = next(multiplier * magnitude for multiplier in [1, 2, 2.5, 5, 10]
                    if multiplier * magnitude >= target_step)
        upper = np.ceil(maximum * 1.05 / step) * step
        settings = dict(range=[0, float(upper)], autorange=False, tickmode='linear',
                        tick0=0, dtick=float(step), showline=True, linecolor='#c7d0cb')
        if axis == 'x':
            fig.update_xaxes(showgrid=False, **settings)
        else:
            fig.update_yaxes(gridcolor='#edf0ef', zeroline=False, **settings)
    st.plotly_chart(fig,width='stretch',key='relationship',config={'displayModeBar':False})

st.divider()
st.header('Compare Housing Indicators', anchor=False)
with st.container(border=True):
    compare_health = 'Asthma ED Visits per 10,000 Youth'
    compare_y = 'edvisr2024'
    compare_sample = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*HOUSING.values(), compare_y])
    st.caption(f'n = {len(compare_sample)} ZIP Code Zones')
    for method in ['Scatterplot matrix', 'Linear association', 'Local regression']:
        st.markdown('**' + display_title(method) + '**')
        fig = matrix_figure(compare_sample, HOUSING, compare_y, compare_health, method)
        st.plotly_chart(fig, width='stretch', key='compare_' + method,
                        config={'displayModeBar': False})


st.divider()
st.header('Where Housing Reports and Asthma Rates Overlap or Cluster', anchor=False)
spatial_view = st.segmented_control(
    'Spatial analysis', ['Co-location', 'Quantile LISA', 'Hot and cold spots'],
    default='Co-location', required=True, format_func=display_title, label_visibility='collapsed', key='spatial_view',
)
spatial_figures = {
    'Co-location': ([15,16], [
        'ZIP Zones Where Asthma and Lead Reporting Rates Fall in the Same Group (Q1–Q6)',
        'ZIP Zones Where Both Rates Fall in the Highest Group (Q6)',
    ]),
    'Quantile LISA': ([17,18,19], [
        'Clusters of High Youth Asthma ED Visit Rates',
        'Clusters of High Lead Inspection Reporting Rates',
        'Clusters Where Both Rates Are High',
    ]),
    'Hot and cold spots': ([20,21], [
        'Hot and Cold Spots in Youth Asthma ED Visit Rates',
        'Hot and Cold Spots in Lead Inspection Reporting Rates',
    ]),
}
if spatial_view == 'Co-location':
    st.write('Youth asthma ED visit rates and lead inspection reporting rates are each divided into six groups, '
             'from lowest (Q1) to highest (Q6). The left map highlights ZIP zones in the same group for both rates, and '
             'the right map highlights only those in Q6 for both. Matching groups indicate similar rankings, not equal rates.')
elif spatial_view == 'Quantile LISA':
    st.write('Tests whether ZIP zones in the highest group (Q6) cluster with neighboring zones, '
             'for asthma, lead reporting, or both together.')
elif spatial_view == 'Hot and cold spots':
    st.write('Local Moran identifies statistically significant hot spots (High–High), cold spots (Low–Low), '
             'and ZIP zones that contrast with their neighbors (spatial outliers: High–Low or Low–High).')
references(*spatial_figures[spatial_view])
