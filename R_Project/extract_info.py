import re
import sys

def debug_print(label, data):
    """Print debug information in a structured format"""
    print(f"\n=== DEBUG: {label} ===", file=sys.stderr)
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {value} (type: {type(value)})", file=sys.stderr)
    else:
        print(f"{data} (type: {type(data)})", file=sys.stderr)
    print("===================\n", file=sys.stderr)

def extract_all_features(text):
    debug_print("Input text", text)
    text = text.lower()
    info = {}

    def extract_value(key):
        debug_print(f"Extracting value for key", key)
        pattern = rf'{key}.*?(\d+(\.\d+)?)'
        match = re.search(pattern, text)
        if match:
            val = match.group(1)
            debug_print(f"Found value for {key}", val)
            try:
                if '.' in val:
                    return float(val)
                else:
                    return int(val)
            except Exception as e:
                debug_print(f"Error converting value for {key}", str(e))
                return val
        else:
            debug_print(f"No value found for {key}", None)
            return None

    # Define expected types and valid ranges for each feature
    feature_specs = {
        'Age': {'type': int, 'range': (0, 120)},
        'Gender': {'type': int, 'range': (0, 1)},  # Binary
        'Ethnicity': {'type': int, 'range': (0, 5)},
        'EducationLevel': {'type': int, 'range': (0, 5)},
        'BMI': {'type': float, 'range': (10, 50)},
        'Smoking': {'type': int, 'range': (0, 1)},  # Binary
        'AlcoholConsumption': {'type': int, 'range': (0, 1)},  # Binary
        'PhysicalActivity': {'type': int, 'range': (0, 5)},
        'DietQuality': {'type': int, 'range': (0, 5)},
        'SleepQuality': {'type': int, 'range': (0, 5)},
        'FamilyHistoryAlzheimers': {'type': int, 'range': (0, 1)},  # Binary
        'CardiovascularDisease': {'type': int, 'range': (0, 1)},  # Binary
        'Diabetes': {'type': int, 'range': (0, 1)},  # Binary
        'Depression': {'type': int, 'range': (0, 1)},  # Binary
        'HeadInjury': {'type': int, 'range': (0, 1)},  # Binary
        'Hypertension': {'type': int, 'range': (0, 1)},  # Binary
        'SystolicBP': {'type': int, 'range': (60, 200)},
        'DiastolicBP': {'type': int, 'range': (40, 120)},
        'CholesterolTotal': {'type': int, 'range': (100, 400)},
        'CholesterolLDL': {'type': int, 'range': (50, 300)},
        'CholesterolHDL': {'type': int, 'range': (20, 100)},
        'CholesterolTriglycerides': {'type': int, 'range': (50, 500)},
        'MMSE': {'type': int, 'range': (0, 30)},
        'FunctionalAssessment': {'type': int, 'range': (0, 10)},
        'MemoryComplaints': {'type': int, 'range': (0, 1)},  # Binary
        'BehavioralProblems': {'type': int, 'range': (0, 1)},  # Binary
        'ADL': {'type': int, 'range': (0, 10)},
        'Confusion': {'type': int, 'range': (0, 1)},  # Binary
        'Disorientation': {'type': int, 'range': (0, 1)},  # Binary
        'PersonalityChanges': {'type': int, 'range': (0, 1)},  # Binary
        'DifficultyCompletingTasks': {'type': int, 'range': (0, 1)},  # Binary
        'Forgetfulness': {'type': int, 'range': (0, 1)}  # Binary
    }

    keys_map = {
        'Age': ['age'],
        'Gender': ['gender', 'sex', 'sexe'],
        'Ethnicity': ['ethnicity', 'ethnie'],
        'EducationLevel': ['educationlevel', 'education level', 'niveau d\'éducation'],
        'BMI': ['bmi', 'body mass index', 'indice de masse corporelle'],
        'Smoking': ['smoking', 'tabagisme'],
        'AlcoholConsumption': ['alcoholconsumption', 'alcohol consumption', 'consommation d\'alcool', 'alcool'],
        'PhysicalActivity': ['physicalactivity', 'physical activity', 'activité physique'],
        'DietQuality': ['dietquality', 'diet quality', 'qualité du régime alimentaire'],
        'SleepQuality': ['sleepquality', 'sleep quality', 'qualité du sommeil'],
        'FamilyHistoryAlzheimers': ['familyhistoryalzheimers', 'family history', 'antécédent familial', 'famille alzheimer'],
        'CardiovascularDisease': ['cardiovasculardisease', 'cardiovascular disease', 'maladies cardiovasculaires'],
        'Diabetes': ['diabetes', 'diabète'],
        'Depression': ['depression', 'dépression'],
        'HeadInjury': ['headinjury', 'head injury', 'traumatisme crânien'],
        'Hypertension': ['hypertension', 'tension'],
        'SystolicBP': ['systolicbp', 'systolic blood pressure', 'pression systolique'],
        'DiastolicBP': ['diastolicbp', 'diastolic blood pressure', 'pression diastolique'],
        'CholesterolTotal': ['cholesteroltotal', 'cholesterol total', 'cholestérol total'],
        'CholesterolLDL': ['cholesteroldld', 'ldl cholesterol', 'cholesterol ldl', 'CholesterolLDL'],
        'CholesterolHDL': ['cholesterolhdl', 'hdl cholesterol', 'cholesterol hdl'],
        'CholesterolTriglycerides': ['cholesteroltriglycerides', 'triglycerides', 'cholesterol triglycerides'],
        'MMSE': ['mmse', 'mini mental state examination'],
        'FunctionalAssessment': ['functionalassessment', 'functional assessment', 'évaluation fonctionnelle'],
        'MemoryComplaints': ['memorycomplaints', 'memory complaints', 'plaintes mnésiques'],
        'BehavioralProblems': ['behavioralproblems', 'behavioral problems', 'problèmes comportementaux'],
        'ADL': ['adl', 'activities of daily living', 'activités de la vie quotidienne'],
        'Confusion': ['confusion'],
        'Disorientation': ['disorientation', 'désorientation'],
        'PersonalityChanges': ['personalitychanges', 'personality changes', 'changements de personnalité'],
        'DifficultyCompletingTasks': ['difficultycompletingtasks', 'difficulty completing tasks', 'difficulté à accomplir les tâches'],
        'Forgetfulness': ['forgetfulness', 'oublis']
    }

    for feature, keywords in keys_map.items():
        debug_print(f"Processing feature", feature)
        val = None
        for key in keywords:
            val = extract_value(key)
            if val is not None:
                debug_print(f"Found value for {feature} using keyword {key}", val)
                break
        
        if val is None:
            debug_print(f"Missing value for feature", feature)
            raise ValueError(f"Missing value for feature '{feature}' in input text!")
        
        # Get feature specifications
        spec = feature_specs[feature]
        expected_type = spec['type']
        min_val, max_val = spec['range']
        
        debug_print(f"Validating {feature}", {
            'value': val,
            'expected_type': expected_type.__name__,
            'valid_range': (min_val, max_val)
        })
        
        try:
            # Convert to expected type
            converted_val = expected_type(val)
            
            # Validate range
            if not (min_val <= converted_val <= max_val):
                debug_print(f"Value out of range for {feature}", {
                    'value': converted_val,
                    'range': (min_val, max_val)
                })
                raise ValueError(f"Value {converted_val} for {feature} is outside valid range [{min_val}, {max_val}]")
            
            info[feature] = converted_val
            debug_print(f"Successfully processed {feature}", {
                'final_value': converted_val,
                'final_type': type(converted_val).__name__
            })
            
        except (ValueError, TypeError) as e:
            debug_print(f"Error processing {feature}", str(e))
            raise ValueError(f"Invalid value for feature '{feature}': {str(e)}")

    debug_print("Final extracted features", info)
    return info
