#define H 3
#define W 2

free env int(1, W) xt;
env int(0, H) yt;

assume env proctype taz(){
    do
    :: yt = yt - 1
    :: yt = yt + 1
    :: skip
    od
}

assume ltl { []<>(yt == 0) }

sys int(0, H) x;
sys int(0, H) y;

assert sys proctype bunny(){
    do
    :: x = x - 1
    :: x = x + 1
    :: y = y - 1
    :: y = y + 1
    :: skip
    od
}

assert ltl {
    [] ! ((x == xt) && (y == yt)) &&
    /* [] ! --X ((xt' == x ) && (yt' == y )) && */
    [] -X ! ((xt' == x) && (yt' == y)) &&
    []<>((x == H) && (y == 0)) &&
    []<>((x == 0) && (y == H)) &&
    []<>((x == 0) && (y == 0)) &&
    []<>((x == H) && (y == H))}
