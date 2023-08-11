def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


s = [16, 24, 32, 64, 96, 160, 320]
for i in s:
    input_channel = _make_divisible(i, 8)
    print(input_channel)
