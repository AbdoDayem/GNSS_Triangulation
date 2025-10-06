from datetime import datetime
from io import TextIOWrapper

from position import Satellite

# Nav File Processing: https://github.com/MarwinMarw/RINEX-satellite-position
# Obs File Processing: https://github.com/IanGoodbody/Rinex_Parser

class EndOfFile(Exception):
    pass

class ErrorOBSRecord(Exception):
    pass

def _split_neg_num(number, start_index=0):
    index_minus = number.find('-', start_index)
    fixed = []

    if index_minus > 0 and not number[index_minus-1].isalpha():
        num1 = number[:index_minus]
        num2 = number[index_minus:]
        fixn1 = _split_neg_num(num1)
        fixn2 = _split_neg_num(num2)
        
        if fixn1 is None:
            fixed.append(num1)
        else:
            for i in fixn1:
                fixed.append(i)

        if fixn2 is None:
            fixed.append(num2)
        else:
            for i in fixn2:
                fixed.append(i)

        return fixed

    else:    
        if index_minus != -1:
            return _split_neg_num(number, index_minus+1)
        else:
            return None

def _fix_negative_num(nums: list) -> list:
    fixed_nums = []
    for num in nums:
        fixed_num = _split_neg_num(num)
        if fixed_num is not None:
            for fn in fixed_num:
                fixed_nums.append(fn)
        else:
            fixed_nums.append(num)

    return fixed_nums

def skip_header(rinex_file: TextIOWrapper) -> int:
    noLine = 0
    while True:
        noLine += 1
        if 'END' in rinex_file.readline():
            break

    return noLine

def read_PRN_EPOCH_SV_CLK(nums: list) -> dict:
    if len(nums) != 10:
        raise ErrorOBSRecord(f'PRN_EPOCH_SV_CLK read error str: {str(nums)}')
    
    return {
        'PRN': nums[0],
        'EPOCH': {
            'YEAR': nums[1],
            'MONTH': nums[2],
            'DAY': nums[3],
            'HOUR': nums[4],
            'MINUTE': nums[5],
            'SECOND': nums[6],
        },
        'SV_clock_bias':float(nums[7]),
        'SV_clock_drift':float(nums[8]),
        'SV_clock_drift_rate':float(nums[9])
    }
    
def read_BROADCAST_ORBIT_1(nums: list) -> dict:
    if len(nums) != 4:
        raise ErrorOBSRecord(f'BROADCAST_ORBIT_1 read error str: {str(nums)}')

    return {
        'IODE': float(nums[0]),
        'Crs': float(nums[1]),
        'Delta_n': float(nums[2]),
        'M0': float(nums[3])
    }

def read_BROADCAST_ORBIT_2(nums: list) -> dict:
    if len(nums) != 4:
        raise ErrorOBSRecord(f'BROADCAST_ORBIT_2 read error str: {str(nums)}')

    return {
        'Cuc': float(nums[0]),
        'e_Eccentricity': float(nums[1]),
        'Cus': float(nums[2]),
        'sqrt_A': float(nums[3])
    }

def read_BROADCAST_ORBIT_3(nums: list) -> dict:
    if len(nums) != 4:
        raise ErrorOBSRecord(f'BROADCAST_ORBIT_3 read error str: {str(nums)}')

    return {
        'Toe': float(nums[0]),
        'Cic': float(nums[1]),
        'OMEGA': float(nums[2]),
        'Cis': float(nums[3])
    }

def read_BROADCAST_ORBIT_4(nums: list) -> dict:
    if len(nums) != 4:
        raise ErrorOBSRecord(f'BROADCAST_ORBIT_4 read error str: {str(nums)}')

    return {
        'i0': float(nums[0]),
        'Crc': float(nums[1]),
        'omega': float(nums[2]),
        'OMEGA_DOT': float(nums[3])
    }

def read_BROADCAST_ORBIT_5(nums: list) -> dict:
    if len(nums) != 4:
        raise ErrorOBSRecord(f'BROADCAST_ORBIT_5 read error str: {str(nums)}')

    return {
        'IDOT': float(nums[0]),
        'Codes_L2_channel': float(nums[1]),
        'GPS_week': float(nums[2]),
        'L2_P': float(nums[3])
    }

def read_BROADCAST_ORBIT_6(nums: list) -> dict:
    if len(nums) != 4:
        raise ErrorOBSRecord(f'BROADCAST_ORBIT_6 read error str: {str(nums)}')

    return {
        'SV_accuracy': float(nums[0]),
        'SV_health': float(nums[1]),
        'TGD': float(nums[2]),
        'IODC': float(nums[3])
    }

def read_BROADCAST_ORBIT_7(nums: list) -> dict:
    if len(nums) != 2:
        raise ErrorOBSRecord(f'BROADCAST_ORBIT_7 read error str: {str(nums)}')

    return {
        'TTM': float(nums[0]),
        'Fit_interval': float(nums[1])
    }

def _next_line(rinex_file: TextIOWrapper) -> list:
    line = rinex_file.readline()
    if not line or line.isspace():
        raise EndOfFile

    nums = [num for num in line.strip().replace('D', 'e').split(' ') if num != '']
    fixed_nums = _fix_negative_num(nums)

    return fixed_nums

def _extract_nav(rinex_file: TextIOWrapper) -> dict:    
    ext_data = {}
    nr_sat = 0
    nr_line = 0
    while True:
        try:
            str_data = []

            for _ in range(8):
                nr_line += 1
                data_from_string = _next_line(rinex_file)
                str_data.append(data_from_string)
                
            ex_data_l1 = read_PRN_EPOCH_SV_CLK(str_data[0])
            key = f"{nr_sat}_{str(ex_data_l1['PRN'])}"
            nr_sat += 1

            ext_data[key] = ex_data_l1
            ext_data[key].update(read_BROADCAST_ORBIT_1(str_data[1]))
            ext_data[key].update(read_BROADCAST_ORBIT_2(str_data[2]))
            ext_data[key].update(read_BROADCAST_ORBIT_3(str_data[3]))
            ext_data[key].update(read_BROADCAST_ORBIT_4(str_data[4]))
            ext_data[key].update(read_BROADCAST_ORBIT_5(str_data[5]))
            ext_data[key].update(read_BROADCAST_ORBIT_6(str_data[6]))
            ext_data[key].update(read_BROADCAST_ORBIT_7(str_data[7]))

        except EndOfFile:
            break
        
        except ErrorOBSRecord as eobsr:
            print(f'Error: OBS Record {nr_sat}, Data: {str_data}, NoLine: {nr_line}', eobsr)
            break
        
    return ext_data

def _extract_obs(rinFile: TextIOWrapper):    
    ext_data = {}
    obsTime = 0
    currentLine = rinFile.readline()
    if currentLine[0:1] == ">":
        year = int(currentLine[2:6])
        month = int(currentLine[7:9])
        day = int(currentLine[10:12])
        hour = int(currentLine[13:15])
        mm = int(currentLine[16:18])
        sec = int(currentLine[19:21])

        obsTime = datetime(year,month,day,hour,mm,sec)
        
        # Parse the PRN values for this epoch
        numSats = int(currentLine[33:35])
        for i in range(numSats):
            currentLine = rinFile.readline()
            prn = currentLine[0:3]
            if prn.startswith("G"):
                # Based on where the L1C pseudorange is located in the observation file
                #pseudo = currentLine[5:17]
                pseudo = currentLine[69:81]
                if not pseudo.isspace():
                    ext_data[prn] = float(pseudo)
    return obsTime, ext_data

def read_nav(filename: str) -> dict:
    ext_data = None
    with open(filename, 'r') as rinex_file:
        skipped_lines = skip_header(rinex_file)
        ext_data = _extract_nav(rinex_file)
    return ext_data

'''
Processes position file
Parameters:
    filename: position file path
Output: 
    rec = {
        obstime: [lat, lon, height]
    }
'''
def get_rec(filename: str) -> dict:
    posfile = open(filename, 'r')
    rec_loc = {}
    while True:
        if 'latitude(deg)' in posfile.readline():
            break
    currentLine = posfile.readline()
    while currentLine.strip() != "":
        year = int(currentLine[0:4])
        month = int(currentLine[5:7])
        day = int(currentLine[8:10])
        hour = int(currentLine[11:13])
        mm = int(currentLine[14:16])
        sec = int(currentLine[17:19])
        obsTime = obsTime = datetime(year,month,day,hour,mm,sec)
        coord = [float(currentLine[26:38]), float(currentLine[40:53]), float(currentLine[57:64])]
        rec_loc[obsTime] = coord
        currentLine = posfile.readline()
    return rec_loc

def get_rec_ascii(filename: str):
    posfile = open(filename, 'r')
    rec_loc = []
    currentLine = posfile.readline()
    while currentLine.strip() != "":
        if not "INSUFFICIENT_OBS" in currentLine:
            if "PPP_CONVERGING" in currentLine:
                lat = float(currentLine[104:118])
                lon = float(currentLine[119:134])
                rec_loc.append([lat, lon])   
            else:
                lat = float(currentLine[93:107])
                lon = float(currentLine[108:123])
                rec_loc.append([lat, lon])
        else: rec_loc.append("NONE")
        currentLine = posfile.readline()
    return rec_loc

'''
Processes next observation in obs file
Parameters:
    sats: set of satellites in the navigation file
    file: observation file
Output: 
    obsTime =  observation time
    for every sat in observation && sats:
        sat.tObs = obsTime
        sat.pseudoRange = L1 C/A PseudoRange
'''
def next_obs(sats, file):
    obsTime, ext_data = _extract_obs(file)
    for sat in sats:
        if sat in ext_data:
            sats[sat].tObs = obsTime
            sats[sat].pseudoRange = ext_data[sat]
    return obsTime

'''
Processes navigation file, opens observation file
Parameters:
    filenav: navigation file path
    fileobs: observation file path
Output: 
    sats = {
        PRN: Satellite object with CEI variable & PRN fields initialized
    }
    obsFile: opened observation file with header skipped
'''
def get_cei(filenav: str, fileobs:str):
    result = read_nav(filenav)
    obsFile = open(fileobs, 'r')
    skip_header(obsFile)
    sats = {}
    for key in result:
        sats[result[key]["PRN"]] = Satellite(PRN = result[key]["PRN"],af0 = result[key]["SV_clock_bias"], af1 = result[key]["SV_clock_drift"], af2 = result[key]["SV_clock_drift_rate"],
                    Tsv = result[key]["TTM"], Toe = result[key]["Toe"], GPSWeek = result[key]["GPS_week"],
                    toc = datetime(int(result[key]["EPOCH"]["YEAR"]),int(result[key]["EPOCH"]["MONTH"]),int(result[key]["EPOCH"]["DAY"]),
                    int(result[key]["EPOCH"]["HOUR"]),int(result[key]["EPOCH"]["MINUTE"]),int(float(result[key]["EPOCH"]["SECOND"]))),
                    e = result[key]["e_Eccentricity"], sqrtA= result[key]["sqrt_A"], Cic= result[key]["Cic"], Crc= result[key]["Crc"],
                    Cis= result[key]["Cis"], Crs= result[key]["Crs"], Cuc= result[key]["Cuc"], Cus= result[key]["Cus"],
                    DeltaN= result[key]["Delta_n"], Omega0= result[key]["OMEGA"], omega=result[key]["omega"], Io=result[key]["i0"],
                    OmegaDot= result[key]["OMEGA_DOT"],IDOT= result[key]["IDOT"], M0= result[key]["M0"], Tgd=result[key]["TGD"])
    return sats, obsFile

