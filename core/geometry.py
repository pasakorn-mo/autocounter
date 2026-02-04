def _side(line, p):
    (x1, y1), (x2, y2) = line
    return (p[0]-x1)*(y2-y1) - (p[1]-y1)*(x2-x1)

def crossed_line(line, p_prev, p_now):
    return _side(line, p_prev) * _side(line, p_now) < 0
