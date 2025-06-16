# location config
DBID = '1-25590'
BEACON_LOCATION = {
	'02-00702': {'loc': (0, 0), 'aspect': 180, 'color': '#B0C4DE'},
	'02-00726': {'loc': (-2.4, 23.2), 'aspect': 90, 'color': '#B0C4DE'},
	'02-00682': {'loc': (25.6, 23.2), 'aspect': 0, 'color': '#B0C4DE'},
	'02-00722': {'loc': (29.6, 14.4), 'aspect': 0, 'color': '#B0C4DE'},
	'02-01931': {'loc': (29.6, 0), 'aspect': 270, 'color': '#B0C4DE'},
}

BEACON_LOCATION_1011 = {
	'01-02728': {'loc': (0, 1.6), 'aspect': 180, 'color': '#B0C4DE'},
	'02-02043': {'loc': (29.6, 0.8), 'aspect': 270, 'color': '#B0C4DE'},
	'02-00580': {'loc': (29.6, 14.4), 'aspect': 0, 'color': '#B0C4DE'},
	'01-02716': {'loc': (25.6, 23.2), 'aspect': 0, 'color': '#B0C4DE'},
	'02-01931': {'loc': (-2.4, 23.2), 'aspect': 90, 'color': '#B0C4DE'},
}

BEACON_LOCATION_1012 = {
	'01-02728': {'loc': (0, 1.6), 'aspect': 180, 'color': '#B0C4DE'},
	'02-02043': {'loc': (29.6, 0.8), 'aspect': 270, 'color': '#B0C4DE'},
	'02-00580': {'loc': (29.6, 14.4), 'aspect': 0, 'color': '#B0C4DE'},
	'01-02716': {'loc': (26.4, 23.2), 'aspect': 0, 'color': '#B0C4DE'},
	'02-01931': {'loc': (-2.4, 23.2), 'aspect': 90, 'color': '#B0C4DE'},
}


AXLIM = -40, 60
FIGSIZE = 20, 10

ALIVE_CIRCLE = dict()
ALLTIME_DATA = dict()


# 权重占比 半径与角度、密集度、历史位置、时间
score_weight = [0.2, 1, 0.1, 0.4]
