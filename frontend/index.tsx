import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { createRoot } from 'react-dom/client';


const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080';

// === TYP-DEFINITIONEN ===
interface LibraryItem {
  id: string;
  type: 'lyric' | 'style' | 'technique' | 'generated_lyric' | 'rhyme_lesson' | 'emphasis' | 'rhyme_flow' | 'rhyme_lesson_group';
  content: string;
  title?: string;
  sourceLyricId?: string;
  emphasisPattern?: string;
  rhymeFlowPattern?: string;
  userEmphasisExplanation?: string;
  userRhymeFlowExplanation?: string;
  structuredData?: any; // NEU
}

interface Profile {
  id: string;
  name: string;
  library: (LibraryItem | RuleCategory | RhymeLessonGroup)[]; // <-- HIER ERWEITERN
  styleClusters?: any[];      // NEU
  techniqueClusters?: any[];  // NEU
}

type View = 'start' | 'transition' | 'analyze' | 'define_style' | 'define_technique' | 'rhyme_machine' | 'write_song' | 'manage_library' | 'trainer' | 'kuenstler_dna';
type LibraryTab = 'learned_lyrics' | 'generated_lyrics' | 'style' | 'technique' | 'emphasis' | 'rhyme_flow' | 'rhyme_lessons';

interface PendingAnalysis {
  lyrics: string;
  title: string;
  performanceStyle: 'sung' | 'rapped' | 'unknown';
  audioBlob?: Blob;
}

// Für eine einzelne, spezifische Regel
interface LearnedRule {
  id: string;
  title: string;
  definition: string;
}

// Für eine Gruppe von Regeln (unsere neue Kategorie)
interface RuleCategory {
  id: string;
  type: 'rule_category'; // Wichtig zur Unterscheidung von anderen Elementen
  categoryTitle: string;
  rules: LearnedRule[];
}

// Ein einzelner Reim innerhalb einer Gruppe
interface RhymeLessonItem {
  id: string;
  rhymingWord: string;
}

// Der "Sammel-Ordner" für ein Ausgangswort
interface RhymeLessonGroup extends LibraryItem {
  type: 'rhyme_lesson_group';
  targetWord: string;
  vowelSequence: string;
  syllableCount: number;
  rhymes: RhymeLessonItem[];
}

// Erweitere den Bibliotheks-Typ in der Profile-Definition

// === SVG ICONS ===
// === ZENTRALES ICON-VERZEICHNIS FÜR PANELS ===
// Verzeichnis Nr. 1: Für die Haupt-Icons (Sidemenü & große Panel-Titel)
const MAIN_ICONS: Record<string, string> = {
    ANALYZE: '🔬',
    STYLE: '🕶️',
    TECHNIQUE: '🎛️',
    TRAINER: '🤖',
    RHYME: '✍🏼',
    WRITE: '👻',
    LIBRARY: '📚',
    DNA: '🧬',
    DEFAULT: 'ℹ️'
};

// Verzeichnis Nr. 2: Für die untergeordneten Info-Box-Icons (SectionHeader)
const INFO_ICONS: Record<string, string> = {
    AUDIO_ANALYSIS: '🎵',
    TEXT_INPUT: '📝',
    RHYME_FINDER: '🔎',
    RHYME_GENERATOR: '💡',
    COMPOSITION: '🎼',
    STYLE_DEFINITION: '🎭',
    TECHNIQUE_DEFINITION: '⚡',
    DEFAULT: '🔹'
};

    
const UI_ICONS: Record<string, JSX.Element> = {
    ADD: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>,
    CLOSE: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>,
    CHEVRON_DOWN: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>,
    
    // NEU: Füge dieses Icon hinzu
    SEND: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>,
    
    // WICHTIG: Ein neues Standard-Icon als Fallback
    DEFAULT: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
};
// === THEMATISCHES ICON-VERZEICHNIS FÜR SECTION-HEADERS ===
// Dieses Objekt dient als unser zentrales Verzeichnis für thematische Icons in Section-Headers
    
  
// === CUSTOM DROPDOWN KOMPONENTE ===
// NEUE WIEDERVERWENDBARE KOMPONENTE
const CustomDropdown = ({ options, selectedValue, onSelect, placeholder }: {
    options: Array<{ id: string; title: string }>;
    selectedValue: string;
    onSelect: (optionId: string) => void;
    placeholder: string;
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Schließt das Dropdown, wenn außerhalb geklickt wird
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [dropdownRef]);
    
    const handleSelect = (optionId: string) => {
        onSelect(optionId);
        setIsOpen(false);
    };

    // Finde den Titel des ausgewählten Songs
    const selectedOption = options.find(option => option.id === selectedValue);
    const displayValue = selectedOption ? selectedOption.title : placeholder;

    return (
        <div className="custom-dropdown" ref={dropdownRef}>
            <button className="dropdown-toggle" onClick={() => setIsOpen(!isOpen)}>
                <span>{displayValue}</span>
                <span className={`dropdown-chevron ${isOpen ? 'open' : ''}`}>
                    {UI_ICONS.CHEVRON_DOWN}
                </span>
            </button>

            {isOpen && (
                <ul className="dropdown-menu">
                    {options.map(option => (
                        <li key={option.id} className="dropdown-item">
                            <button onClick={() => handleSelect(option.id)}>
                                {option.title}
                            </button>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};

// NEUE MULTI-SELECT DROPDOWN KOMPONENTE
const MultiSelectDropdown = ({ options, selectedValues, onToggleOption, placeholder }: {
    options: Array<{ id: string; content: string }>;
    selectedValues: string[];
    onToggleOption: (optionId: string) => void;
    placeholder: string;
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Schließt das Dropdown, wenn außerhalb geklickt wird
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [dropdownRef]);

    const displayValue = selectedValues.length > 0 
        ? `${selectedValues.length} Element(e) ausgewählt` 
        : placeholder;

    return (
        <div className="custom-dropdown" ref={dropdownRef}>
            <button className="dropdown-toggle" onClick={() => setIsOpen(!isOpen)}>
                <span>{displayValue}</span>
                <span className={`dropdown-chevron ${isOpen ? 'open' : ''}`}>
                    {UI_ICONS.CHEVRON_DOWN}
                </span>
            </button>

            {isOpen && (
                <ul className="dropdown-menu">
                    {options.map(option => (
                        <li key={option.id} className="multi-select-item">
                            <label className="checkbox-label">
                                <input 
                                    type="checkbox"
                                    checked={selectedValues.includes(option.id)}
                                    onChange={() => onToggleOption(option.id)}
                                />
                                <span>{option.content}</span>
                            </label>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};

// NEUE KOMPONENTE FÜR DAS REIM-MODAL
const RhymeModal = ({ isOpen, onClose, targetWord, onSelectRhyme, getKnowledgeBase }) => {
    if (!isOpen) return null;

    const [rhymeInput, setRhymeInput] = useState(targetWord);
    const [rhymeResults, setRhymeResults] = useState([]);
    const [isFindingRhymes, setIsFindingRhymes] = useState(false);
    const [error, setError] = useState('');

    // Startet die Reimsuche, wenn das Modal geöffnet wird
    useEffect(() => {
        setRhymeInput(targetWord);
        if (targetWord) {
            handleFindRhymes();
        }
    }, [targetWord]);

    const handleFindRhymes = async () => {
        if (!rhymeInput.trim()) return;
        setIsFindingRhymes(true);
        setError('');
        setRhymeResults([]);
        try {
            const dnaResult = getKnowledgeBase();
            const rhymePayload = {
                input: rhymeInput,
                knowledgeBase: dnaResult.success ? dnaResult.knowledgeBase : null,
                max_words: 8
            };
            const response = await fetch(`${BACKEND_URL}/api/rhymes`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(rhymePayload),
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Reim-Suche fehlgeschlagen');
            }
            const result = await response.json();
            setRhymeResults(result.rhymes || []);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsFindingRhymes(false);
        }
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <SectionHeader
                    iconKey="RHYME_RESULTS"
                    title={`Reime für "${targetWord}"`}
                    description="Wähle einen passenden Reim aus der Liste aus."
                />
                <div className="rhyme-step-container">
                    <input
                        type="text"
                        value={rhymeInput}
                        onChange={e => setRhymeInput(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleFindRhymes()}
                        className="form-input-field"
                    />
                    <button onClick={handleFindRhymes} disabled={isFindingRhymes} className="main-action-button">
                        {isFindingRhymes ? 'Suche...' : 'Suchen'}
                    </button>
                </div>
                <div className="rhyme-modal-results">
                    {error && <p className="error-message">{error}</p>}
                    {rhymeResults.length > 0 ? (
                        rhymeResults.map((result, index) => (
                            <button key={index} className="rhyme-result-button" onClick={() => onSelectRhyme(result.rhyme)}>
                                {result.rhyme}
                            </button>
                        ))
                    ) : (
                        !isFindingRhymes && <p>Keine Reime gefunden.</p>
                    )}
                </div>
            </div>
        </div>
    );
};

// === WIEDERVERWENDBARE SECTION-HEADER-KOMPONENTE ===
const SectionHeader = ({ iconKey, title, description }: { iconKey: keyof typeof INFO_ICONS; title: string; description: string }) => {
  // Die Komponente schlägt im INFO_ICONS-Verzeichnis nach.
  // Findet sie nichts, nimmt sie das Standard-Icon.
  const icon = INFO_ICONS[iconKey] || INFO_ICONS.DEFAULT;

  return (
    <div className="section-header">
      <div className="header-icon">{icon}</div>
      <div className="header-content">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
};

const LoadingOverlay = ({ message }: { message: string }) => (
    <div className="loading-overlay-central">
        <div className="loading-spinner"></div>
        <p>{message}</p>
    </div>
);

const ChatInput = React.memo(({ onSend, isReplying }: { onSend: (text: string) => void; isReplying: boolean; }) => {
    const [text, setText] = useState('');

    const handleSend = () => {
        if (!text.trim()) return;
        onSend(text);
        setText('');
    };

    return (
        <div className="chat-input-area">
            <textarea
                placeholder="Bringe der KI etwas bei..."
                value={text}
                onChange={e => setText(e.target.value)}
                onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSend();
                    }
                }}
                disabled={isReplying}
            />
            <button 
                className="main-action-button chat-send-button icon-button" /* <-- Neue Klasse für das Styling */
                onClick={handleSend} 
                disabled={isReplying || !text.trim()}
                title="Senden" /* <-- Wichtig für Barrierefreiheit */
            >
                {UI_ICONS.SEND} {/* <-- Hier kommt das Icon rein */}
            </button>
        </div>
    );
});

const MemoizedInput = React.memo(({
    value,
    onChange,
    placeholder,
}: {
    value: string;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    placeholder: string;
}) => {
    return (
        <input
            type="text"
            placeholder={placeholder}
            value={value}
            onChange={onChange}
        />
    );
});

const MemoizedTextarea = React.memo(({
    value,
    onChange,
    placeholder,
    className, // 1. className als Prop empfangen
}: {
    value: string;
    onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
    placeholder: string;
    className?: string; // Prop-Typen ebenfalls anpassen
}) => {
    return (
        <textarea
            placeholder={placeholder}
            value={value}
            onChange={onChange}
            className={className} // 2. Die empfangene className hier anwenden
        />
    );
});

    const Panel: React.FC<{ title: string, description: string, iconKey: string, children: React.ReactNode }> = ({ title, description, iconKey, children }) => {
        // Sucht das Icon im Verzeichnis oder nimmt das Standard-Icon
        const icon = MAIN_ICONS[iconKey] || MAIN_ICONS.DEFAULT;

        return (
            <div className="panel-container">
                <div className="panel-header">
                    <div className="panel-header-title">
                <div className="panel-title-icon">{icon}</div>
                        <h2>{title}</h2>
                    </div>
                    {/* Der Profil-Wahlschalter wird hier komplett entfernt */}
                </div>
                <p className="panel-description">{description}</p>
                <div className="panel-content">
                    {children}
                </div>
            </div>
        );
    };


// === HAUPTKOMPONENTE: App ===
const App = () => {
    // === STATE MANAGEMENT ===
    const [userId, setUserId] = useState<string | null>(null);
    const [profiles, setProfiles] = useState<Record<string, Profile>>({});
    const [activeProfileId, setActiveProfileId] = useState<string | null>(null);
    const [currentView, setCurrentView] = useState<View>('start');
    const [isStarting, setIsStarting] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');
    const [statusMessage, setStatusMessage] = useState({ text: '', isError: false });
    const [isLyricEditorModalOpen, setIsLyricEditorModalOpen] = useState(false);
    const [isTitleModalOpen, setIsTitleModalOpen] = useState(false);
    const [editableLyrics, setEditableLyrics] = useState('');
    const [editableTitle, setEditableTitle] = useState('');
    const [pendingAnalysis, setPendingAnalysis] = useState<PendingAnalysis | null>(null);
    const [libraryTab, setLibraryTab] = useState<LibraryTab>('learned_lyrics');
    const [openLyricId, setOpenLyricId] = useState<string | null>(null);
    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [isCreateProfileModalOpen, setIsCreateProfileModalOpen] = useState(false);
    const [isTransferProfileModalOpen, setIsTransferProfileModalOpen] = useState(false);
    const [transferSourceProfileId, setTransferSourceProfileId] = useState<string>('');
    const [transferTargetProfileId, setTransferTargetProfileId] = useState<string>('');
    const [editingItem, setEditingItem] = useState<LibraryItem | null>(null);
    const [editedContent, setEditedContent] = useState('');
    const [feedbackText, setFeedbackText] = useState('');
    const [manualTitle, setManualTitle] = useState('');
    const [manualLyrics, setManualLyrics] = useState('');
    const [rhymeInput, setRhymeInput] = useState('');
    const [multiRhymeInput, setMultiRhymeInput] = useState('');
    const [numLinesToGenerate, setNumLinesToGenerate] = useState(7);
    const [lessonWord, setLessonWord] = useState('');
    const [lessonRhyme, setLessonRhyme] = useState('');


    const [songStyles, setSongStyles] = useState<string[]>([]);
    const [songTechniques, setSongTechniques] = useState<string[]>([]);
    const [songBeatDescription, setSongBeatDescription] = useState('');
    const [songBPM, setSongBPM] = useState('');
    const [songKey, setSongKey] = useState('');
    const [songPerformanceStyle, setSongPerformanceStyle] = useState('Gerappt');
    const [songTopic, setSongTopic] = useState('');
    const [songPartType, setSongPartType] = useState('full_song');
    const [numLines, setNumLines] = useState(16);

    // Handler-Funktionen für Multi-Select
    const handleToggleStyle = (styleId: string) => {
        setSongStyles(prev => 
            prev.includes(styleId) 
                ? prev.filter(id => id !== styleId) 
                : [...prev, styleId]
        );
    };

    const handleToggleTechnique = (techniqueId: string) => {
        setSongTechniques(prev => 
            prev.includes(techniqueId) 
                ? prev.filter(id => id !== techniqueId) 
                : [...prev, techniqueId]
        );
    };
    const [newProfileName, setNewProfileName] = useState('');
    const [rhymeResults, setRhymeResults] = useState<any[]>([]);
    const [isFindingRhymes, setIsFindingRhymes] = useState(false);
    const [multiRhymeResults, setMultiRhymeResults] = useState<any[]>([]);

    const [isGeneratingMultiRhymes, setIsGeneratingMultiRhymes] = useState(false);
    const [syllableCount, setSyllableCount] = useState(0);
    const [vowelSequence, setVowelSequence] = useState('');
    const [trainerMessages, setTrainerMessages] = useState<{ text: string, isUser: boolean }[]>([]);
    const [trainerInput, setTrainerInput] = useState('');
    const [isTrainerReplying, setIsTrainerReplying] = useState(false);
    const [openRuleCategoryId, setOpenRuleCategoryId] = useState<string | null>(null);
    const [searchPerformed, setSearchPerformed] = useState(false);
    const [isRuleEditModalOpen, setIsRuleEditModalOpen] = useState(false);
    const [editingRule, setEditingRule] = useState<{ categoryId: string, rule: LearnedRule } | null>(null);
    const [editedRuleContent, setEditedRuleContent] = useState('');
    const [isEditingCategoryTitle, setIsEditingCategoryTitle] = useState<{ id: string, title: string } | null>(null);
    const [isEditingRuleTitle, setIsEditingRuleTitle] = useState<{ categoryId: string, ruleId: string, title: string } | null>(null);
    const [isEditingRuleDefinition, setIsEditingRuleDefinition] = useState<{ categoryId: string, ruleId: string, definition: string } | null>(null);
    const [isSubRhymeEditModalOpen, setIsSubRhymeEditModalOpen] = useState(false);
    const [editingSubRhyme, setEditingSubRhyme] = useState<{ groupId: string, rhyme: RhymeLessonItem } | null>(null);
    const [editedSubRhymeContent, setEditedSubRhymeContent] = useState('');
    const [generatedSongText, setGeneratedSongText] = useState('');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisAbortController, setAnalysisAbortController] = useState<AbortController | null>(null);
    const [activeDnaItemIds, setActiveDnaItemIds] = useState<string[]>([]);
    const [pendingDnaSelection, setPendingDnaSelection] = useState<Record<string, boolean>>({});
    const [isSynthesizing, setIsSynthesizing] = useState(false);
    const [styleViewMode, setStyleViewMode] = useState<'single' | 'grouped'>('single');
    const [openClusterIds, setOpenClusterIds] = useState<Record<string, boolean>>({});
    const [isSynthesizingTechniques, setIsSynthesizingTechniques] = useState(false);
    const [techniqueViewMode, setTechniqueViewMode] = useState<'single' | 'grouped'>('single');

    // NEU: State für das Bestätigungs-Modal
    const [confirmationState, setConfirmationState] = useState<{
        isOpen: boolean;
        message: string;
        details?: string;
        actionKey?: string;
        onConfirm?: () => void;
    }>({ isOpen: false, message: '' });

    // NEU: State für "Nicht mehr nachfragen" im Bestätigungs-Modal
    const [dontAskAgain, setDontAskAgain] = useState(false);

    // NEU: State für die Optimierung von Bibliotheks-Einträgen
    const [optimizingItemId, setOptimizingItemId] = useState<string | null>(null);
    const [openTechniqueClusterIds, setOpenTechniqueClusterIds] = useState<Record<string, boolean>>({});
    const [openDnaCategoryKey, setOpenDnaCategoryKey] = useState<string | null>(null);

    // Neue State-Variablen für das Rhyme-Modal
    const [isRhymeModalOpen, setIsRhymeModalOpen] = useState(false);
    const [rhymeModalWord, setRhymeModalWord] = useState('');

    // Neue State-Variablen für die DNA-Reim-Verwaltung

    // NEU: States und Refs für die Aufnahme
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const [activeRhymeDnaIds, setActiveRhymeDnaIds] = useState<string[]>([]);
    const [rhymeFilterKeyword, setRhymeFilterKeyword] = useState('');

    // Refs
    const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const beatFileInputRef = useRef<HTMLInputElement | null>(null);

    // NEU: State für die Detailansicht von Betonung/Reimfluss
    const [detailViewMode, setDetailViewMode] = useState<'summary' | 'full'>('summary');

    // === PERSISTENZ & KERNLOGIK ===
    useEffect(() => {
        let id = localStorage.getItem('lyricMachineUserId');
        if (!id) {
            id = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            localStorage.setItem('lyricMachineUserId', id);
        }
        setUserId(id);
    }, []);

    useEffect(() => {
        if (!userId) return;
        const savedState = localStorage.getItem(`lyricMachineState_${userId}`);
        if (savedState) {
            try {
                const state = JSON.parse(savedState);
                setProfiles(state.profiles || {});
                setActiveProfileId(state.activeProfileId);
                setActiveDnaItemIds(state.activeDnaItemIds || []);
            } catch (e) { console.error("Failed to parse state:", e); createDefaultProfile(); }
        } else { createDefaultProfile(); }
    }, [userId]);

    const createDefaultProfile = () => {
        const defaultId = `profile-${Date.now()}`;
        const defaultProfile = { id: defaultId, name: 'Default Profile', library: [] };
        setProfiles({ [defaultId]: defaultProfile });
        setActiveProfileId(defaultId);
    };

    useEffect(() => {
        if (!userId || !activeProfileId || Object.keys(profiles).length === 0) return;
        if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
        saveTimeoutRef.current = setTimeout(() => {
            const stateToSave = { profiles, activeProfileId, activeDnaItemIds };
            localStorage.setItem(`lyricMachineState_${userId}`, JSON.stringify(stateToSave));
        }, 1500);
        return () => { if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current); };
    }, [profiles, activeProfileId, activeDnaItemIds, userId]);

    const activeProfile = activeProfileId ? profiles[activeProfileId] : null;

    // Füge dies oben in deiner App-Komponente ein
    useEffect(() => {
        console.log("Technik-Cluster haben sich geändert:", activeProfile?.techniqueClusters);
    }, [activeProfile]);

    const groupedItems = useMemo(() => {
        if (!activeProfile || (libraryTab !== 'style' && libraryTab !== 'technique')) return null;
        
        const lyrics = activeProfile.library.filter(item => item.type === 'lyric');
        const itemType = libraryTab === 'style' ? 'style' : 'technique';
        const itemsToGroup = activeProfile.library.filter(item => item.type === itemType);

        // Sortiere die Song-Gruppen alphabetisch nach Songtitel
        const sortedLyrics = [...lyrics].sort((a, b) => (a.title || '').localeCompare(b.title || ''));

        return sortedLyrics.map(lyric => ({
            lyric,
            // HIER KOMMT DIE NEUE SORTIERUNG FÜR DIE EINZELNEN EINTRÄGE
            items: itemsToGroup
                .filter(item => item.sourceLyricId === lyric.id)
                .sort((a, b) => a.content.localeCompare(b.content))
        })).filter(group => group.items.length > 0);
    }, [activeProfile, libraryTab]);

    // === HANDLER-FUNKTIONEN ===

    // NEU: Funktion, die das Bestätigungs-Modal aufruft
    const requestConfirmation = (config: { message: string; details?: string; actionKey: string; onConfirm: () => void; }) => {
        const { message, details, actionKey, onConfirm } = config;

        // Prüfen, ob der Nutzer diese Abfrage deaktiviert hat
        if (localStorage.getItem(`confirm_${actionKey}`) === 'true') {
            onConfirm(); // Aktion direkt ausführen
            return;
        }
        
        // Andernfalls, das Modal anzeigen
        setConfirmationState({
            isOpen: true,
            message,
            details,
            actionKey,
            onConfirm
        });
    };

    // NEU: Funktion für die Optimierung von Bibliotheks-Einträgen für die DNA
    const handleOptimizeForDna = async (itemId: string) => {
        if (!activeProfile) return;
        setOptimizingItemId(itemId); // Ladezustand für diesen spezifischen Button aktivieren

        const itemToOptimize = activeProfile.library.find(item => item.id === itemId);
        if (!itemToOptimize) {
            showStatus("Element nicht gefunden.", true);
            setOptimizingItemId(null);
            return;
        }

        try {
            const response = await fetch(`${BACKEND_URL}/api/structure-analysis`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    analysisText: itemToOptimize.content,
                    analysisType: itemToOptimize.type // 'emphasis' oder 'rhyme_flow'
                }),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Optimierung fehlgeschlagen');
            }

            const result = await response.json();
            
            // Update das LibraryItem mit den neuen strukturierten Daten
            updateProfileLibrary(lib => lib.map(item =>
                item.id === itemId ? { ...item, structuredData: result.structuredData } : item
            ), "Optimierung für DNA erfolgreich!");

        } catch (error: any) {
            showStatus(error, true);
        } finally {
            setOptimizingItemId(null); // Ladezustand deaktivieren
        }
    };
    
    // NEU: Wandelt strukturierte Daten in eine lesbare Zusammenfassung um
    const generateReadableSummary = (item: LibraryItem) => {
        if (!item.structuredData) {
            return item.content; // Fallback für nicht-optimierte Elemente
        }

        const data = item.structuredData;
        let summary = '';

        if (item.type === 'emphasis') {
            summary = `Muster: ${data.pattern_type || 'unbekannt'}. Fokus: ${data.main_focus || 'unbekannt'}. Merkmale: ${(data.key_features || []).join(', ')}.`;
        } else if (item.type === 'rhyme_flow') {
            summary = `Schema: ${data.scheme || 'unbekannt'}. Flow: ${data.flow_type || 'unbekannt'}. Merkmale: ${(data.features || []).join(', ')}.`;
        } else {
            return item.content; // Fallback für andere Typen
        }
        
        return summary;
    };
    
    // NEU: Funktion zum Starten und Stoppen der Audio-Aufnahme
    const handleToggleRecording = async () => {
        if (isRecording) {
            // Fall 1: Aufnahme läuft bereits -> Aufnahme stoppen
            mediaRecorderRef.current?.stop();
            setIsRecording(false);
            showStatus("Aufnahme beendet. Starte Analyse...", false);
        } else {
            // Fall 2: Keine Aufnahme aktiv -> Aufnahme starten
            try {
                // Zugriff auf das Mikrofon anfordern
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // MediaRecorder-Instanz erstellen
                const mediaRecorder = new MediaRecorder(stream);
                mediaRecorderRef.current = mediaRecorder;
                
                // Audiodaten-Array für diese Aufnahme zurücksetzen
                audioChunksRef.current = [];

                // Event-Handler, der aufgerufen wird, wenn Audiodaten verfügbar sind
                mediaRecorder.ondataavailable = (event) => {
                    audioChunksRef.current.push(event.data);
                };

                // Event-Handler, der nach dem Stoppen der Aufnahme ausgeführt wird
                mediaRecorder.onstop = () => {
                    // Erstelle ein einzelnes Blob-Objekt aus den gesammelten Chunks
                    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                    
                    // Nutze die bereits existierende Funktion zur Analyse!
                    handleInitialAudioAnalysis(audioBlob, 'audio/webm');
                    
                    // Stream-Tracks stoppen, um das Mikrofon-Icon im Browser zu deaktivieren
                    stream.getTracks().forEach(track => track.stop());
                };

                // Aufnahme starten und State aktualisieren
                mediaRecorder.start();
                setIsRecording(true);
                showStatus("Aufnahme gestartet...", false);

            } catch (err) {
                console.error("Fehler beim Zugriff auf das Mikrofon:", err);
                showStatus("Zugriff auf das Mikrofon fehlgeschlagen. Bitte Berechtigung erteilen.", true);
            }
        }
    };

    const showStatus = (messageOrError: string | Error, isError: boolean = false) => {
        let textToShow = '';
        let isAnError = isError;

        if (messageOrError instanceof Error) {
            isAnError = true;
            const errorMessage = messageOrError.message.toLowerCase();

            // Hier ist unser "Fehler-Übersetzer"
            if (errorMessage.includes('failed to fetch')) {
                textToShow = 'Verbindung zum Server fehlgeschlagen. Bitte prüfe deine Internetverbindung.';
            } else if (errorMessage.includes('overloaded') || errorMessage.includes('529')) {
                textToShow = 'Die KI ist momentan stark ausgelastet. Bitte versuche es in ein paar Momenten erneut.';
            } else {
                // Ein allgemeiner Fallback für andere technische Fehler
                textToShow = 'Ein unerwarteter Fehler ist aufgetreten. Bitte versuche es später erneut.';
            }
        } else {
            textToShow = messageOrError;
        }

        setStatusMessage({ text: textToShow, isError: isAnError });
        setTimeout(() => setStatusMessage({ text: '', isError: false }), 5000);
    };

    const updateProfileLibrary = useCallback((updateFn: (library: LibraryItem[]) => LibraryItem[] , successMessage?: string) => {
        if (!activeProfileId) return;
        setProfiles(prev => {
            const currentProfile = prev[activeProfileId];
            if (!currentProfile) return prev;
            return { ...prev, [activeProfileId]: { ...currentProfile, library: updateFn([...currentProfile.library]) } };
        });
        if (successMessage) {
            showStatus(successMessage, false);
        }
    }, [activeProfileId, showStatus]);

    const handleCreateProfile = () => {
        if (!newProfileName.trim()) {
            showStatus("Profilname darf nicht leer sein.", true);
            return;
        }
        const newId = `profile-${Date.now()}`;
        const newProfile: Profile = { id: newId, name: newProfileName, library: [] };
        setProfiles(prev => ({ ...prev, [newId]: newProfile }));
        setActiveProfileId(newId);
        setNewProfileName('');
        setIsCreateProfileModalOpen(false);
        showStatus(`Profil '${newProfileName}' wurde erstellt.`, false);
    };

    const handleTransferProfile = () => {
        if (!transferSourceProfileId || !transferTargetProfileId) {
            showStatus("Bitte wähle Quell- und Zielprofil aus.", true);
            return;
        }
        
        if (transferSourceProfileId === transferTargetProfileId) {
            showStatus("Quell- und Zielprofil müssen unterschiedlich sein.", true);
            return;
        }

        const sourceProfile = profiles[transferSourceProfileId];
        const targetProfile = profiles[transferTargetProfileId];
        
        if (!sourceProfile || !targetProfile) {
            showStatus("Ein oder beide Profile wurden nicht gefunden.", true);
            return;
        }

        // Alle Daten vom Quellprofil auf das Zielprofil übertragen
        const updatedTargetProfile = {
            ...targetProfile,
            library: [...targetProfile.library, ...sourceProfile.library]
        };

        // Quellprofil leeren (alle Daten wurden übertragen)
        const updatedSourceProfile = {
            ...sourceProfile,
            library: []
        };

        // Profile aktualisieren
        setProfiles(prev => ({
            ...prev,
            [transferSourceProfileId]: updatedSourceProfile,
            [transferTargetProfileId]: updatedTargetProfile
        }));

        // Modal schließen und Status anzeigen
        setIsTransferProfileModalOpen(false);
        setTransferSourceProfileId('');
        setTransferTargetProfileId('');
        
        const sourceName = sourceProfile.name;
        const targetName = targetProfile.name;
        showStatus(`Alle Daten von '${sourceName}' wurden erfolgreich auf '${targetName}' übertragen.`, false);
    };

    const handleDeleteItem = (itemIdToDelete: string) => {
        if (!window.confirm("Sind Sie sicher, dass Sie diesen Eintrag und alle zugehörigen Elemente löschen möchten?")) return;
        if (!activeProfileId) return;

        setProfiles(prev => {
            const profileToUpdate = prev[activeProfileId];
            if (!profileToUpdate) return prev;

            // Finde das zu löschende Element, egal wo es sich befindet
            let itemToDelete = profileToUpdate.library.find(i => i.id === itemIdToDelete);
            
            // WICHTIG: Falls es in der Hauptbibliothek nicht gefunden wurde (weil es nur noch im Cluster existiert),
            // suchen wir es in den Clustern, um seinen Typ zu bestimmen.
            if (!itemToDelete) {
                for (const cluster of [...(profileToUpdate.styleClusters || []), ...(profileToUpdate.techniqueClusters || [])]) {
                    const foundFacet = cluster.facets.find((facet: any) => facet.id === itemIdToDelete);
                    if (foundFacet) {
                        // Wir nehmen an, der Typ entspricht dem Tab, um die Logik zu vervollständigen
                        itemToDelete = { ...foundFacet, type: libraryTab }; 
                        break;
                    }
                }
            }
            
            if (!itemToDelete) return prev; // Sicherheitshalber, falls nichts gefunden wird

            const itemType = itemToDelete.type;

            // Schritt 1: Filtere IMMER die Haupt-Bibliothek
            const newLibrary = itemType === 'lyric'
                ? profileToUpdate.library.filter(item => item.id !== itemIdToDelete && item.sourceLyricId !== itemIdToDelete)
                : profileToUpdate.library.filter(item => item.id !== itemIdToDelete);

            // Schritt 2: Filtere IMMER die Cluster-Arrays
            let newStyleClusters = profileToUpdate.styleClusters;
            if (itemType === 'style' && profileToUpdate.styleClusters) {
                newStyleClusters = profileToUpdate.styleClusters
                    .map(cluster => ({
                        ...cluster,
                        facets: cluster.facets.filter((facet: any) => facet.id !== itemIdToDelete),
                    }))
                    .filter(cluster => cluster.facets.length > 0);
            }
            
            let newTechniqueClusters = profileToUpdate.techniqueClusters;
            if (itemType === 'technique' && profileToUpdate.techniqueClusters) {
                newTechniqueClusters = profileToUpdate.techniqueClusters
                    .map(cluster => ({
                        ...cluster,
                        facets: cluster.facets.filter((facet: any) => facet.id !== itemIdToDelete),
                    }))
                    .filter(cluster => cluster.facets.length > 0);
            }

            // Schritt 3: Baue das komplett neue Profil-Objekt zusammen
            const updatedProfile = {
                ...profileToUpdate,
                library: newLibrary,
                styleClusters: newStyleClusters,
                techniqueClusters: newTechniqueClusters,
            };

            return {
                ...prev,
                [activeProfileId]: updatedProfile,
            };
        });

        showStatus("Eintrag erfolgreich gelöscht.", false);
    };

    const handleStartEditing = (item: LibraryItem) => {
        setEditingItem(item);
        setEditedContent(item.content);
        setIsEditModalOpen(true);
    };

    const handleSaveEdit = () => {
        if (!editingItem) return;
        updateProfileLibrary(lib => lib.map(item =>
            item.id === editingItem.id ? { ...item, content: editedContent } : item
        ), "Änderung gespeichert.");
        setIsEditModalOpen(false);
        setEditingItem(null);
        setEditedContent('');
    };

    // === DNA-MANAGEMENT HANDLER ===
    const handleUpdateActiveDna = (itemIds: string[]) => {
        setActiveDnaItemIds(prev => {
            const newIds = [...prev];
            itemIds.forEach(id => {
                if (!newIds.includes(id)) {
                    newIds.push(id);
                }
            });
            return newIds;
        });
    };

    const handleRemoveFromActiveDna = (itemId: string) => {
        setActiveDnaItemIds(prev => prev.filter(id => id !== itemId));
    };

    const handleClearActiveDna = () => {
        setActiveDnaItemIds([]);
    };

    const handleSaveFeedback = (lyricId: string, feedbackType: 'emphasis' | 'rhymeFlow') => {
        const key = feedbackType === 'emphasis' ? 'userEmphasisExplanation' : 'userRhymeFlowExplanation';
        updateProfileLibrary(lib => lib.map(item =>
            item.id === lyricId ? { ...item, [key]: feedbackText } : item
        ), "Feedback gespeichert. Es wird bei der nächsten Analyse berücksichtigt.");
        setFeedbackText('');
    };

    // === RHYME-MODAL HANDLER ===
    const handleTextSelection = () => {
        const selectedText = window.getSelection()?.toString().trim();
        if (selectedText && selectedText.length > 2) {
            setRhymeModalWord(selectedText);
            setIsRhymeModalOpen(true);
        }
    };

    const handleSelectRhyme = (newRhyme: string) => {
        // Ersetzt das ursprünglich markierte Wort durch den neuen Reim
        setGeneratedSongText(prev => prev.replace(rhymeModalWord, newRhyme));
        setIsRhymeModalOpen(false);
    };

    // === DNA-REIM-VERWALTUNG HANDLER ===
    const handleAddRecentRhymesToDna = () => {
        // 1. Finde alle gelernten Reim-Gruppen
        const allRhymeGroups = activeProfile?.library.filter(
            item => item.type === 'rhyme_lesson_group'
        ) as RhymeLessonGroup[];

        if (!allRhymeGroups || allRhymeGroups.length === 0) {
            showStatus("Keine gelernten Reime zum Hinzufügen gefunden.", true);
            return;
        }

        // 2. Extrahiere alle einzelnen Reime und nimm die letzten 20
        const allRhymes = allRhymeGroups.flatMap(group => group.rhymes.map(rhyme => rhyme.id));
        const recentRhymeIds = allRhymes.slice(-20);

        // 3. Aktualisiere den State
        setActiveRhymeDnaIds(recentRhymeIds);
        showStatus(`Die letzten ${recentRhymeIds.length} Reime wurden zur DNA hinzugefügt.`, false);
    };

    const handleClearRhymesFromDna = () => {
        setActiveRhymeDnaIds([]);
        showStatus("Alle Reim-Beispiele wurden aus der aktiven DNA entfernt.", false);
    };

    const handleAddFilteredRhymesToDna = () => {
        if (!rhymeFilterKeyword.trim()) return;

        // 1. Finde alle Reime, die das Schlüsselwort enthalten
        const allRhymeGroups = activeProfile?.library.filter(
            item => item.type === 'rhyme_lesson_group'
        ) as RhymeLessonGroup[];

        const filteredRhymeIds = allRhymeGroups.flatMap(group => 
            group.rhymes.filter(rhyme => 
                group.targetWord.toLowerCase().includes(rhymeFilterKeyword.toLowerCase()) || 
                rhyme.rhymingWord.toLowerCase().includes(rhymeFilterKeyword.toLowerCase())
        ).map(rhyme => rhyme.id)
        );

        if (filteredRhymeIds.length === 0) {
            showStatus(`Keine Reime passend zum Begriff "${rhymeFilterKeyword}" gefunden.`, true);
            return;
        }

        // 2. Nimm maximal 20 der gefilterten Reime
        const limitedFilteredIds = filteredRhymeIds.slice(0, 20);
        
        // 3. Aktualisiere den State
        setActiveRhymeDnaIds(limitedFilteredIds);
        showStatus(`${limitedFilteredIds.length} gefilterte Reime wurden zur DNA hinzugefügt.`, false);
    };

    const handleAnalyzeWord = () => {
    if (!rhymeInput.trim()) {
        showStatus("Bitte gib ein Wort ein.", true);
        return;
    }
    setSearchPerformed(false);
    const analysis = getPhoneticBreakdown(rhymeInput);
    setSyllableCount(analysis.syllableCount);
    setVowelSequence(analysis.vowelSequence);
    setRhymeResults([]); // Alte Ergebnisse löschen
};

    const handleGenerateMultiRhymes = async () => {
        if (!multiRhymeInput.trim()) {
            showStatus("Bitte gib eine Eingabezeile ein.", true);
            return;
        }
        
        // Erstelle einen neuen AbortController für diese Analyse
        const abortController = new AbortController();
        setAnalysisAbortController(abortController);
        setIsGeneratingMultiRhymes(true);
        setLoadingMessage('Generiere Reimzeilen...');
        
        // Warte kurz, damit der UI-Update sichtbar wird
        await new Promise(resolve => setTimeout(resolve, 100));

        try {
            // Prüfe, ob Künstler-DNA verfügbar ist, aber blockiere nicht
            const dnaResult = getUnifiedKnowledgeBase();
            const knowledgeBase = dnaResult.success ? dnaResult.knowledgeBase : null;

            const multiRhymePayload = {
                input_line: multiRhymeInput,  // Korrigiert: input_line statt inputLine
                knowledge_base: knowledgeBase, // Korrigiert: knowledge_base statt knowledgeBase
                num_lines: numLinesToGenerate // Fordere z.B. 7 Zeilen an
            };

            const response = await fetch(`${BACKEND_URL}/api/generate-rhyme-line`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(multiRhymePayload),
                signal: abortController.signal
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Reimzeilen-Generierung fehlgeschlagen');
            }

            const result = await response.json();
            
            // Korrigiert: Verwende result.generated_lines statt result.rhymes
            if (result.generated_lines && Array.isArray(result.generated_lines)) {
                // Konvertiere die generierten Zeilen in das erwartete Format
                const formattedResults = (result.generated_lines || []).map((item: { line: string; analysis: any }) => ({
                    line: item.line,
                    explanation: `Phonetische Resonanz: ${item.analysis.overall_similarity_score}/10 | Vokal-Rhythmus Match: ${item.analysis.vowel_rhythm_match ? '✓' : '✗'}`
                }));
                
                setMultiRhymeResults(formattedResults);
                showStatus(`Reimzeilen erfolgreich generiert! ${result.generated_lines.length} Folgezeilen erstellt.`, false);
            } else {
                throw new Error('Unerwartetes Antwortformat vom Server');
            }

        } catch (error: any) {
            if (error.name === 'AbortError') {
                showStatus('Reimzeilen-Generierung wurde abgebrochen.', false);
            } else {
                showStatus(error, true);
                setMultiRhymeResults([]); // Bei Fehler Ergebnisliste leeren
            }
        } finally {
            setLoadingMessage('');
            setIsGeneratingMultiRhymes(false);
            setAnalysisAbortController(null);
        }
    };



    const handleFindRhymesWithAnalysis = async () => {
    if (!rhymeInput.trim()) {
        showStatus("Bitte gib ein Wort ein.", true);
        return;
    }
    
    // Erstelle einen neuen AbortController für diese Analyse
    const abortController = new AbortController();
    setAnalysisAbortController(abortController);
    setIsFindingRhymes(true);
    setSearchPerformed(false); // Wichtig: Alten Suchstatus zurücksetzen
    
    // Warte kurz, damit der UI-Update sichtbar wird
    await new Promise(resolve => setTimeout(resolve, 100));

    try {
        // Prüfe, ob Künstler-DNA verfügbar ist, aber blockiere nicht
        const dnaResult = getUnifiedKnowledgeBase();
        const knowledgeBase = dnaResult.success ? dnaResult.knowledgeBase : null;

        const rhymePayload = {
            input: rhymeInput,
            knowledgeBase: knowledgeBase, // Kann null sein, falls keine DNA vorhanden
            max_words: 8
        };

        const rhymeResponse = await fetch(`${BACKEND_URL}/api/rhymes`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rhymePayload),
            signal: abortController.signal
        });

        if (!rhymeResponse.ok) {
            const err = await rhymeResponse.json();
            throw new Error(err.error || 'Reim-Suche fehlgeschlagen');
        }

        const rhymeResult = await rhymeResponse.json();
        // NEU: Filtere ungültige Einträge heraus, bevor der State gesetzt wird
        const validRhymes = (rhymeResult.rhymes || []).filter(r => 
            r.rhyme && !r.rhyme.toUpperCase().includes('AUSGESCHLOSSEN')
        );
        setRhymeResults(validRhymes);

    } catch (error: any) {
        if (error.name === 'AbortError') {
            showStatus('Reim-Suche wurde abgebrochen.', false);
        } else {
            showStatus(error, true);
            setRhymeResults([]); // Bei Fehler Ergebnisliste leeren
        }
    } finally {
        setIsFindingRhymes(false);
        setSearchPerformed(true); // Nach jedem Versuch (erfolgreich oder nicht) Suche als durchgeführt markieren
        setAnalysisAbortController(null);
    }
}; 

const handleDeleteRuleCategory = useCallback((categoryId: string) => {
    if (!window.confirm("Sicher, dass du diese gesamte Regelkategorie löschen möchtest?")) return;
    updateProfileLibrary(lib => lib.filter(item => item.id !== categoryId), "Kategorie gelöscht.");
}, [updateProfileLibrary]);

const handleDeleteRule = useCallback((categoryId: string, ruleId: string) => {
    if (!window.confirm("Sicher, dass du diese Regel löschen möchtest?")) return;
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === categoryId && item.type === 'rule_category') {
            return { ...item, rules: item.rules.filter(rule => rule.id !== ruleId) };
        }
        return item;
    }), "Regel gelöscht.");
}, [updateProfileLibrary]);

const handleDeleteSubRhyme = (groupId: string, rhymeId: string) => {
    if (!window.confirm("Diesen einzelnen Reim löschen?")) return;
    updateProfileLibrary(lib => {
        const updatedLib = lib.map(item => {
            if (item.id === groupId && item.type === 'rhyme_lesson_group') {
                const group = item as RhymeLessonGroup;
                const updatedRhymes = group.rhymes.filter(r => r.id !== rhymeId);
                if (updatedRhymes.length === 0) return null; // Gruppe wird später entfernt
                return { ...group, rhymes: updatedRhymes };
            }
            return item;
        });
        return updatedLib.filter(Boolean) as (LibraryItem | RuleCategory | RhymeLessonGroup)[];
    }, "Reim gelöscht.");
};

// NEUE FUNKTIONEN FÜR DEN BEARBEITEN-BUTTON
const handleStartSubRhymeEdit = (groupId: string, rhyme: RhymeLessonItem) => {
    setEditingSubRhyme({ groupId, rhyme });
    setEditedSubRhymeContent(rhyme.rhymingWord);
    setIsSubRhymeEditModalOpen(true);
};

const handleSaveSubRhymeEdit = () => {
    if (!editingSubRhyme) return;
    const { groupId, rhyme } = editingSubRhyme;
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === groupId && item.type === 'rhyme_lesson_group') {
            const group = item as RhymeLessonGroup;
            const updatedRhymes = group.rhymes.map(r => r.id === rhyme.id ? { ...r, rhymingWord: editedSubRhymeContent } : r);
            return { ...group, rhymes: updatedRhymes };
        }
        return item;
    }), "Reim bearbeitet.");
    setIsSubRhymeEditModalOpen(false);
};

const handleStartRuleEdit = useCallback((categoryId: string, rule: LearnedRule) => {
    setEditingRule({ categoryId, rule });
    setEditedRuleContent(rule.definition);
    setIsRuleEditModalOpen(true);
}, []);

const handleSaveRuleEdit = useCallback(() => {
    if (!editingRule) return;
    const { categoryId, rule } = editingRule;
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === categoryId && item.type === 'rule_category') {
            const updatedRules = item.rules.map(r => r.id === rule.id ? { ...r, definition: editedRuleContent } : r);
            return { ...item, rules: updatedRules };
        }
        return item;
    }), "Regel gespeichert.");
    setIsRuleEditModalOpen(false);
    setEditingRule(null);
    setEditedRuleContent('');
}, [editingRule, editedRuleContent, updateProfileLibrary]);

const handleSaveRhymeLesson = () => {
    if (!lessonWord.trim() || !lessonRhyme.trim()) {
        showStatus("Bitte fülle beide Felder aus.", true);
        return;
    }

    // Teile die Eingabe am Komma und entferne leere Einträge
    const rhymesToSave = lessonRhyme.split(',').map(r => r.trim()).filter(r => r);

    if (rhymesToSave.length === 0) {
        showStatus("Keine gültigen Reime zur Speicherung gefunden.", true);
        return;
    }

    updateProfileLibrary(library => {
        let currentLibrary = [...library];
        
        // Führe die Speicherlogik für jeden einzelnen Reim aus
        rhymesToSave.forEach(rhyme => {
            const existingGroup = currentLibrary.find(item => item.type === 'rhyme_lesson_group' && (item as RhymeLessonGroup).targetWord === lessonWord) as RhymeLessonGroup | undefined;

            if (existingGroup) {
                // FALL 1: Gruppe existiert -> Füge neuen Reim hinzu, falls er noch nicht existiert
                if (!existingGroup.rhymes.some(r => r.rhymingWord === rhyme)) {
                    const newRhyme: RhymeLessonItem = { id: `subrhyme-${Date.now()}`, rhymingWord: rhyme };
                    const updatedGroup = { ...existingGroup, rhymes: [...existingGroup.rhymes, newRhyme] };
                    currentLibrary = currentLibrary.map(item => item.id === existingGroup.id ? updatedGroup : item);
                }
            } else {
                // FALL 2: Gruppe existiert nicht -> Erstelle neue Gruppe
                const analysis = getPhoneticBreakdown(lessonWord);
                const newGroup: RhymeLessonGroup = {
                    id: `rlg-${Date.now()}`,
                    type: 'rhyme_lesson_group',
                    content: `Reim-Lektionen für '${lessonWord}'`,
                    targetWord: lessonWord,
                    vowelSequence: analysis.vowelSequence,
                    syllableCount: analysis.syllableCount,
                    rhymes: [{ id: `subrhyme-${Date.now()}`, rhymingWord: rhyme }]
                };
                currentLibrary.push(newGroup);
            }
        });

        return currentLibrary;

    }, "Reim-Lektion(en) erfolgreich gespeichert.");

    // Felder nach dem Speichern leeren
    setLessonWord('');
    setLessonRhyme('');
};

const handleSaveSingleRhyme = (targetWord: string, newRhyme: string) => {
    updateProfileLibrary(library => {
        let currentLibrary = [...library];
        const existingGroup = currentLibrary.find(item => item.type === 'rhyme_lesson_group' && (item as RhymeLessonGroup).targetWord === targetWord) as RhymeLessonGroup | undefined;

        if (existingGroup) {
            // Gruppe existiert: Füge neuen Reim hinzu, falls noch nicht vorhanden
            if (!existingGroup.rhymes.some(r => r.rhymingWord === newRhyme)) {
                const newRhymeItem: RhymeLessonItem = { id: `subrhyme-${Date.now()}-${Math.random()}`, rhymingWord: newRhyme };
                const updatedGroup = { ...existingGroup, rhymes: [...existingGroup.rhymes, newRhymeItem] };
                return currentLibrary.map(item => item.id === existingGroup.id ? updatedGroup : item);
            }
        } else {
            // Gruppe existiert nicht: Erstelle eine neue Gruppe
            const analysis = getPhoneticBreakdown(targetWord);
            const newGroup: RhymeLessonGroup = {
                id: `rlg-${Date.now()}`,
                type: 'rhyme_lesson_group',
                content: `Reim-Lektionen für '${targetWord}'`,
                targetWord: targetWord,
                vowelSequence: analysis.vowelSequence,
                syllableCount: analysis.syllableCount,
                rhymes: [{ id: `subrhyme-${Date.now()}-${Math.random()}`, rhymingWord: newRhyme }]
            };
            currentLibrary.push(newGroup);
        }
        return currentLibrary;
    }, `Reim '${newRhyme}' wurde zur Bibliothek hinzugefügt.`);
};

const handleEditCategoryTitle = (categoryId: string, newTitle: string) => {
    if (!newTitle.trim()) {
        showStatus("Kategorie-Titel darf nicht leer sein.", true);
        // Optional: Reset auf alten Titel, hier aber erstmal einfach
        setIsEditingCategoryTitle(null);
        return;
    }
    updateProfileLibrary(lib => lib.map(item =>
        (item.id === categoryId && item.type === 'rule_category') ? { ...item, categoryTitle: newTitle } : item
    ));
    setIsEditingCategoryTitle(null);
};

const handleEditRuleTitle = (categoryId: string, ruleId: string, newTitle: string) => {
    if (!newTitle.trim()) {
        showStatus("Regel-Titel darf nicht leer sein.", true);
        setIsEditingRuleTitle(null);
        return;
    }
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === categoryId && item.type === 'rule_category') {
            const updatedRules = item.rules.map(r => r.id === ruleId ? { ...r, title: newTitle } : r);
            return { ...item, rules: updatedRules };
        }
        return item;
    }));
    setIsEditingRuleTitle(null);
};

const handleEditRuleDefinition = (categoryId: string, ruleId: string, newDefinition: string) => {
    if (!newDefinition.trim()) {
        showStatus("Regel-Definition darf nicht leer sein.", true);
        setIsEditingRuleDefinition(null);
        return;
    }
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === categoryId && item.type === 'rule_category') {
            const updatedRules = item.rules.map(r => r.id === ruleId ? { ...r, definition: newDefinition } : r);
            return { ...item, rules: updatedRules };
        }
        return item;
    }));
    setIsEditingRuleDefinition(null);
};

const getUnifiedKnowledgeBase = useCallback(() => {
    if (!activeProfile) {
        return { 
            success: false, 
            error: "Kein aktives Profil ausgewählt. Bitte wähle ein Profil aus, um deine Künstler-DNA zu verwenden." 
        };
    }

    // Prüfe, ob eine manuelle DNA-Auswahl existiert
    if (activeDnaItemIds.length === 0) {
        return { 
            success: false, 
            error: "Du hast noch keine Künstler-DNA für diese Aktion ausgewählt. Bitte stelle zuerst deine DNA im Menü 'Künstler DNA' zusammen." 
        };
    }

    // Wenn eine manuelle Auswahl existiert, filtere die Bibliothek
    const activeIdsSet = new Set(activeDnaItemIds);
    const sourceLibrary = activeProfile.library.filter(item => activeIdsSet.has(item.id));

    // Lade relevante Teile aus der gefilterten Bibliothek
    const ruleCategories = sourceLibrary.filter(item => item.type === 'rule_category') as RuleCategory[];
    const styles = sourceLibrary.filter(item => item.type === 'style');
    const techniques = sourceLibrary.filter(item => item.type === 'technique');
    // NEU: Wir verwenden keine rhymeLessons mehr, sondern rhyme_lesson_group
    const userExplanations = sourceLibrary
        .filter(item => item.type === 'lyric' && (item.userEmphasisExplanation || item.userRhymeFlowExplanation))
        .map(item => item.userEmphasisExplanation || item.userRhymeFlowExplanation);

    let knowledge = "### KÜNSTLER-DNA ###\n" +
                    "Dies ist das gesamte Wissen über den Künstler. Es besteht aus explizit gelernten Regeln, Stil- und Technik-Definitionen sowie Lektionen.\n\n";

    // Baue die Wissensbasis dynamisch aus den gelernten Regeln auf
    if (ruleCategories.length > 0) {
        knowledge += "### GELERNTES REGELWERK (Absolute Wahrheit) ###\n";
        ruleCategories.forEach(category => {
            knowledge += `**Kategorie: ${category.categoryTitle}**\n`;
            category.rules.forEach(rule => {
                knowledge += `- **Regel: ${rule.title}**\n  - Definition: ${rule.definition}\n`;
            });
        });
        knowledge += "\n";
    } else {
        knowledge += "### GELERNTES REGELWERK ###\n- Es wurden noch keine spezifischen Regeln durch den KI-Trainer definiert.\n\n";
    }

    if (styles.length > 0) {
        knowledge += "### KÜNSTLERISCHER STIL & THEMEN ###\n" + styles.map(s => `- ${s.content}`).join("\n") + "\n\n";
    }
    if (techniques.length > 0) {
        knowledge += "### TECHNISCHE FÄHIGKEITEN & VORLIEBEN ###\n" + techniques.map(t => `- ${t.content}`).join("\n") + "\n\n";
    }
    
    // NEU: Berücksichtige nur die aktiv zur DNA hinzugefügten Reime
    const activeRhymeIdsSet = new Set(activeRhymeDnaIds);
    const learnedRhymeGroups = activeProfile.library.filter(
        item => item.type === 'rhyme_lesson_group'
    ) as RhymeLessonGroup[];

    // Baue den `knowledge`-String mit den aktiven Reimen auf
    if (activeRhymeIdsSet.size > 0 && learnedRhymeGroups.length > 0) {
        knowledge += "### EXPLIZITE REIM-LEKTIONEN (Vom Nutzer ausgewählte Beispiele für perfekte Reime) ###\n";
        
        learnedRhymeGroups.forEach(group => {
            // Prüfe, ob die Gruppe rhymes hat und ob es ein Array ist
            if (group.rhymes && Array.isArray(group.rhymes)) {
                group.rhymes.forEach(rhyme => {
                    // Füge den Reim nur hinzu, wenn seine ID im aktiven Set ist
                    if (activeRhymeIdsSet.has(rhyme.id)) {
                        knowledge += `- "${group.targetWord}" reimt sich perfekt auf "${rhyme.rhymingWord}".\n`;
                    }
                });
            }
        });
        knowledge += "\n";
    }
    
    if (userExplanations.length > 0) {
        knowledge += "### NUTZER-KORRIGIERTE EINSICHTEN ###\n" + userExplanations.filter(Boolean).map(e => `- ${e}`).join("\n") + "\n\n";
    }

    // NEU: Verarbeite Betonung und Reimfluss (strukturiert oder als Text)
    const emphasisItems = sourceLibrary.filter(item => item.type === 'emphasis');
    const rhymeFlowItems = sourceLibrary.filter(item => item.type === 'rhyme_flow');

    if (emphasisItems.length > 0) {
        knowledge += "### Betonungsmuster ###\n";
        emphasisItems.forEach(item => {
            if (item.structuredData) {
                knowledge += `- ${JSON.stringify(item.structuredData)}\n`;
            } else {
                knowledge += `- ${item.content}\n`; // Fallback auf alten Text
            }
        });
        knowledge += "\n";
    }

    if (rhymeFlowItems.length > 0) {
        knowledge += "### Reimfluss-Muster ###\n";
        rhymeFlowItems.forEach(item => {
            if (item.structuredData) {
                knowledge += `- ${JSON.stringify(item.structuredData)}\n`;
            } else {
                knowledge += `- ${item.content}\n`; // Fallback auf alten Text
            }
        });
        knowledge += "\n";
    }

    return { success: true, knowledgeBase: knowledge };
}, [activeProfile, activeDnaItemIds, activeRhymeDnaIds]);

// Spezielle Version für Textanalyse ohne Künstler-DNA-Überprüfung
const getUnifiedKnowledgeBaseForTextAnalysis = useCallback(() => {
    if (!activeProfile) {
        return { 
            success: false, 
            error: "Kein aktives Profil ausgewählt. Bitte wähle ein Profil aus." 
        };
    }

    // Für Textanalyse verwenden wir die gesamte Bibliothek, nicht nur die ausgewählte DNA
    const sourceLibrary = activeProfile.library;

    // Lade relevante Teile aus der gesamten Bibliothek
    const ruleCategories = sourceLibrary.filter(item => item.type === 'rule_category') as RuleCategory[];
    const styles = sourceLibrary.filter(item => item.type === 'style');
    const techniques = sourceLibrary.filter(item => item.type === 'technique');
    const rhymeLessons = sourceLibrary.filter(item => item.type === 'rhyme_lesson');
    const userExplanations = sourceLibrary
        .filter(item => item.type === 'lyric' && (item.userEmphasisExplanation || item.userRhymeFlowExplanation))
        .map(item => item.userEmphasisExplanation || item.userRhymeFlowExplanation);

    let knowledge = "### VERFÜGBARES WISSEN ###\n" +
                    "Dies ist das gesamte verfügbare Wissen aus der Bibliothek für die Textanalyse.\n\n";

    // Baue die Wissensbasis dynamisch aus den gelernten Regeln auf
    if (ruleCategories.length > 0) {
        knowledge += "### GELERNTES REGELWERK (Absolute Wahrheit) ###\n";
        ruleCategories.forEach(category => {
            knowledge += `**Kategorie: ${category.categoryTitle}**\n`;
            category.rules.forEach(rule => {
                knowledge += `- **Regel: ${rule.title}**\n  - Definition: ${rule.definition}\n`;
            });
        });
        knowledge += "\n";
    } else {
        knowledge += "### GELERNTES REGELWERK ###\n- Es wurden noch keine spezifischen Regeln durch den KI-Trainer definiert.\n\n";
    }

    if (styles.length > 0) {
        knowledge += "### KÜNSTLERISCHER STIL & THEMEN ###\n" + styles.map(s => `- ${s.content}`).join("\n") + "\n\n";
    }
    if (techniques.length > 0) {
        knowledge += "### TECHNISCHE FÄHIGKEITEN & VORLIEBEN ###\n" + techniques.map(t => `- ${t.content}`).join("\n") + "\n\n";
    }
    if (rhymeLessons.length > 0) {
        knowledge += "### EXPLIZITE REIM-LEKTIONEN (Nutzer-definiert) ###\n" + rhymeLessons.map(l => `- ${l.content}`).join("\n") + "\n\n";
    }
    if (userExplanations.length > 0) {
        knowledge += "### NUTZER-KORRIGIERTE EINSICHTEN ###\n" + userExplanations.filter(Boolean).map(e => `- ${e}`).join("\n") + "\n\n";
    }

    // NEU: Verarbeite Betonung und Reimfluss (strukturiert oder als Text)
    const emphasisItems = sourceLibrary.filter(item => item.type === 'emphasis');
    const rhymeFlowItems = sourceLibrary.filter(item => item.type === 'rhyme_flow');

    if (emphasisItems.length > 0) {
        knowledge += "### Betonungsmuster ###\n";
        emphasisItems.forEach(item => {
            if (item.structuredData) {
                knowledge += `- ${JSON.stringify(item.structuredData)}\n`;
            } else {
                knowledge += `- ${item.content}\n`; // Fallback auf alten Text
            }
        });
        knowledge += "\n";
    }

    if (rhymeFlowItems.length > 0) {
        knowledge += "### Reimfluss-Muster ###\n";
        rhymeFlowItems.forEach(item => {
            if (item.structuredData) {
                knowledge += `- ${JSON.stringify(item.structuredData)}\n`;
            } else {
                knowledge += `- ${item.content}\n`; // Fallback auf alten Text
            }
        });
        knowledge += "\n";
    }

    return { success: true, knowledgeBase: knowledge };
}, [activeProfile]);

const extractThematics = (text: string): string[] => {
    const thematics: string[] = [];
    const lowerText = text.toLowerCase();
    
    // Thematische Kategorien basierend auf Schlüsselwörtern
    const thematicCategories = {
        'natur': ['sonne', 'mond', 'sterne', 'wolken', 'regen', 'wind', 'bäume', 'blumen', 'meer', 'berge'],
        'emotionen': ['liebe', 'schmerz', 'freude', 'trauer', 'wut', 'hoffnung', 'angst', 'glück', 'einsamkeit'],
        'stadtleben': ['straße', 'stadt', 'verkehr', 'menschen', 'leben', 'arbeit', 'party', 'nacht', 'morgen'],
        'sport': ['fußball', 'kicken', 'bolzer', 'spiel', 'gewinnen', 'verlieren', 'team', 'champion'],
        'zeit': ['morgen', 'abend', 'nacht', 'tag', 'woche', 'monat', 'jahr', 'stunde', 'minute'],
        'reisen': ['reise', 'weg', 'ziel', 'start', 'ankunft', 'abfahrt', 'freiheit', 'abenteuer'],
        'musik': ['rhythmus', 'beat', 'melodie', 'klang', 'töne', 'instrument', 'konzert', 'bühne']
    };
    
    // Prüfe jede thematische Kategorie
    Object.entries(thematicCategories).forEach(([category, keywords]) => {
        const foundKeywords = keywords.filter(keyword => lowerText.includes(keyword));
        if (foundKeywords.length > 0) {
            thematics.push(`${category}: ${foundKeywords.join(', ')}`);
        }
    });
    
    // Fallback: Generische Thematik basierend auf Wortlänge und Komplexität
    if (thematics.length === 0) {
        const wordCount = text.split(' ').length;
        if (wordCount <= 3) {
            thematics.push('einfach: Kurze, prägnante Aussage');
        } else if (wordCount <= 6) {
            thematics.push('mittel: Ausführlichere Beschreibung');
        } else {
            thematics.push('komplex: Detaillierte, narrative Struktur');
        }
    }
    
    return thematics;
};

const getPhoneticBreakdown = (word: string): { syllableCount: number; vowelSequence: string } => {
    let tempWord = word.toLowerCase();

    if (tempWord.endsWith('er')) {
        tempWord = tempWord.slice(0, -2) + 'a';
    }    
    
    // Schritt 1: Dehnungs-h nach einem Vokal entfernen
    const vowelsWithH = ['ah', 'eh', 'ih', 'oh', 'uh', 'äh', 'öh', 'üh'];
    vowelsWithH.forEach(vh => {
        tempWord = tempWord.replace(new RegExp(vh, 'g'), vh[0]);
        tempWord = tempWord.replace(/([aouäöü])h/g, '$1'); // Dehnungs-h entfernen (z.B. "mahlen" -> "malen")
        tempWord = tempWord.replace(/ch/g, 'X'); // ch als eigener Konsonant, um nicht 'c' und 'h' einzeln zu werten
        tempWord = tempWord.replace(/sch/g, 'Y'); // sch als eigener Konsonant
        tempWord = tempWord.replace(/ck/g, 'k'); // ck zu k
    });

    // Schritt 2: Diphthonge und Vokale in der richtigen Reihenfolge suchen
    const vowelsAndDiphthongs = ['au', 'eu', 'äu', 'ei', 'ai', 'ey', 'ay', 'ie', 'a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü'];
    const vowelSequenceList: string[] = [];
    let i = 0;
    while (i < tempWord.length) {
        let found = false;
        for (const vOrD of vowelsAndDiphthongs) {
            if (tempWord.substring(i).startsWith(vOrD)) {
                vowelSequenceList.push(vOrD);
                i += vOrD.length;
                found = true;
                break;
            }
        }
        if (!found) {
            i++;
        }
    }
         return {
        syllableCount: vowelSequenceList.length,
        vowelSequence: vowelSequenceList.join('-')
    };
};
    // === VERARBEITUNGSKETTEN ===
    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            handleInitialAudioAnalysis(file, file.type);
        }
    };

    const handleInitialAudioAnalysis = async (blob: Blob, mimeType: string) => {
        // Erstelle einen neuen AbortController für diese Analyse
        const abortController = new AbortController();
        setAnalysisAbortController(abortController);
        setIsAnalyzing(true);
        setLoadingMessage('Transkribiere Audio & führe Voranalyse durch...');
        
        // Warte kurz, damit der UI-Update sichtbar wird
        await new Promise(resolve => setTimeout(resolve, 100));
        
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = async () => {
            const base64Audio = (reader.result as string).split(',')[1];
            try {
                const response = await fetch(`${BACKEND_URL}/api/analyze-audio`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ base64Audio, mimeType }),
                    signal: abortController.signal
                });
                if (!response.ok) { const err = await response.json(); throw new Error(err.error || 'Vor-Analyse fehlgeschlagen'); }
                const result: PendingAnalysis = await response.json();
                setPendingAnalysis({ ...result, audioBlob: blob });
                setEditableTitle(result.title);
                setIsTitleModalOpen(true);
            } catch (error: any) {
                if (error.name === 'AbortError') {
                    showStatus('Audio-Analyse wurde abgebrochen.', false);
                } else {
                    showStatus(error, true);
                }
            } finally {
                setLoadingMessage('');
                setIsAnalyzing(false);
                setAnalysisAbortController(null);
            }
        };
    };

    const handleConfirmTitle = () => {
        if (!pendingAnalysis) return;
        setPendingAnalysis(prev => prev ? { ...prev, title: editableTitle } : null);
        setIsTitleModalOpen(false);
        setEditableLyrics(pendingAnalysis.lyrics);
        setIsLyricEditorModalOpen(true);
    };

    const handleConfirmEditedLyrics = async () => {
        if (!pendingAnalysis || !activeProfileId) return;
        setIsLyricEditorModalOpen(false);
        
        // Erstelle einen neuen AbortController für diese Analyse
        const abortController = new AbortController();
        setAnalysisAbortController(abortController);
        setIsAnalyzing(true);
        setLoadingMessage('Starte umfassende Tiefenanalyse...');
        
        // Warte kurz, damit der UI-Update sichtbar wird
        await new Promise(resolve => setTimeout(resolve, 100));
        
        try {
            // Die Guard Clause und der Aufruf von getUnifiedKnowledgeBase() werden hier nicht mehr benötigt.
            
            // 1. Sammle die bereits existierenden Inhalte
            const existingStyles = activeProfile?.library
                .filter(item => item.type === 'style')
                .map(item => item.content) || [];

            const existingTechniques = activeProfile?.library
                .filter(item => item.type === 'technique')
                .map(item => item.content) || [];

            // 2. Füge sie dem Payload hinzu
            const payload = { 
                lyrics: editableLyrics, 
                // knowledgeBase: dnaResult.knowledgeBase, // ENTFERNT
                hasAudio: !!pendingAnalysis.audioBlob,
                title: pendingAnalysis.title,
                // NEUER TEIL:
                existingLibrary: {
                    styles: existingStyles,
                    techniques: existingTechniques
                }
            };
            const response = await fetch(`${BACKEND_URL}/api/deep-analyze`, {
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify(payload),
                signal: abortController.signal
            });
            if (!response.ok) { const err = await response.json(); throw new Error(err.error || 'Tiefenanalyse fehlgeschlagen'); }
            const result = await response.json();

            // Extrahiere die Analysedaten aus der neuen Struktur
            const analysisData = result.data;
            
            const newLyricItem: LibraryItem = {
                id: `lyric-${Date.now()}`, 
                type: 'lyric', 
                title: result.sourceTitle,
                content: analysisData.formattedLyrics
                // emphasisPattern und rhymeFlowPattern werden nicht mehr direkt gespeichert
            };
            
            const newStyleItems: LibraryItem[] = (analysisData.characterTraits || []).map((trait: string, i: number) => ({ 
                id: `style-${Date.now()}-${i}`, 
                type: 'style', 
                content: trait, 
                sourceLyricId: newLyricItem.id 
            }));
            
            const newTechniqueItems: LibraryItem[] = (analysisData.technicalSkills || []).map((skill: string, i: number) => ({ 
                id: `technique-${Date.now()}-${i}`, 
                type: 'technique', 
                content: skill, 
                sourceLyricId: newLyricItem.id 
            }));

            // Erstelle neue Einträge für Betonung und Reimfluss (konsistent mit handleAnalyzeManualText)
            const newEmphasisItem: LibraryItem = {
                id: `emphasis-${Date.now()}`, 
                type: 'emphasis', 
                content: analysisData.emphasisPattern, 
                sourceLyricId: newLyricItem.id 
            };
            
            const newRhymeFlowItem: LibraryItem = {
                id: `rhyme_flow-${Date.now()}`, 
                type: 'rhyme_flow', 
                content: analysisData.rhymeFlowPattern, 
                sourceLyricId: newLyricItem.id 
            };

            updateProfileLibrary(lib => [...lib, newLyricItem, ...newStyleItems, ...newTechniqueItems, newEmphasisItem, newRhymeFlowItem]);
            showStatus('Analyse erfolgreich abgeschlossen!', false);
            setCurrentView('manage_library');
        } catch (error: any) {
            if (error.name === 'AbortError') {
                showStatus('Tiefenanalyse wurde abgebrochen.', false);
            } else {
                showStatus(error, true);
            }
        } finally {
            setLoadingMessage('');
            setIsAnalyzing(false);
            setAnalysisAbortController(null);
            setPendingAnalysis(null);
        }
    };

    const handleAbortAnalysis = () => {
        if (analysisAbortController) {
            analysisAbortController.abort();
            setLoadingMessage('');
            setIsAnalyzing(false);
            setIsFindingRhymes(false);
            setIsTrainerReplying(false);
            setAnalysisAbortController(null);
        }
    };

    const handleAnalyzeManualText = async () => {
        if (!manualLyrics.trim()) {
            showStatus("Bitte gib einen Songtext ein.", true);
            return;
        }
        
        if (!activeProfileId) {
            showStatus("Kein aktives Profil ausgewählt.", true);
            return;
        }

        // Erstelle einen neuen AbortController für diese Analyse
        const abortController = new AbortController();
        setAnalysisAbortController(abortController);
        setIsAnalyzing(true);
        setLoadingMessage('Analysiere manuell eingegebenen Text...');
        
        // Warte kurz, damit der UI-Update sichtbar wird
        await new Promise(resolve => setTimeout(resolve, 100));
        
        try {
            // Die Guard Clause und der Aufruf von getUnifiedKnowledgeBaseForTextAnalysis() werden hier nicht mehr benötigt.

            // 1. Sammle die bereits existierenden Inhalte
            const existingStyles = activeProfile?.library
                .filter(item => item.type === 'style')
                .map(item => item.content) || [];

            const existingTechniques = activeProfile?.library
                .filter(item => item.type === 'technique')
                .map(item => item.content) || [];

            // 2. Füge sie dem Payload hinzu
            const payload = { 
                lyrics: manualLyrics, 
                // knowledgeBase: dnaResult.knowledgeBase, // ENTFERNT
                hasAudio: false,
                title: manualTitle.trim() || 'Manuell eingegebener Text',
                // NEUER TEIL:
                existingLibrary: {
                    styles: existingStyles,
                    techniques: existingTechniques
                }
            };
            
            const response = await fetch(`${BACKEND_URL}/api/deep-analyze`, {
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify(payload),
                signal: abortController.signal
            });
            
            if (!response.ok) { 
                const err = await response.json(); 
                throw new Error(err.error || 'Textanalyse fehlgeschlagen'); 
            }
            
            const result = await response.json();

            // Extrahiere die Analysedaten aus der neuen Struktur
            const analysisData = result.data;

            // Erstelle neue Bibliothekseinträge
            const newLyricItem: LibraryItem = {
                id: `lyric-${Date.now()}`, 
                type: 'lyric', 
                title: result.sourceTitle,
                content: analysisData.formattedLyrics
                // emphasisPattern und rhymeFlowPattern werden nicht mehr direkt gespeichert
            };
            
            const newStyleItems: LibraryItem[] = (analysisData.characterTraits || []).map((trait: string, i: number) => ({ 
                id: `style-${Date.now()}-${i}`, 
                type: 'style', 
                content: trait, 
                sourceLyricId: newLyricItem.id 
            }));
            
            const newTechniqueItems: LibraryItem[] = (analysisData.technicalSkills || []).map((skill: string, i: number) => ({ 
                id: `technique-${Date.now()}-${i}`, 
                type: 'technique', 
                content: skill, 
                sourceLyricId: newLyricItem.id 
            }));

            // Erstelle neue Einträge für Betonung und Reimfluss
            const newEmphasisItem: LibraryItem = {
                id: `emphasis-${Date.now()}`, 
                type: 'emphasis', 
                content: analysisData.emphasisPattern, 
                sourceLyricId: newLyricItem.id 
            };
            
            const newRhymeFlowItem: LibraryItem = {
                id: `rhyme_flow-${Date.now()}`, 
                type: 'rhyme_flow', 
                content: analysisData.rhymeFlowPattern, 
                sourceLyricId: newLyricItem.id 
            };

            // Füge alle neuen Einträge zur Bibliothek hinzu
            updateProfileLibrary(lib => [...lib, newLyricItem, ...newStyleItems, ...newTechniqueItems, newEmphasisItem, newRhymeFlowItem]);
            
            // Erfolgsmeldung und Weiterleitung zur Bibliothek
            showStatus('Textanalyse erfolgreich abgeschlossen!', false);
            setCurrentView('manage_library');
            
            // Eingabefelder zurücksetzen
            setManualTitle('');
            setManualLyrics('');
            
        } catch (error: any) {
            if (error.name === 'AbortError') {
                showStatus('Analyse wurde abgebrochen.', false);
            } else {
                showStatus(error, true);
            }
        } finally {
            setLoadingMessage('');
            setIsAnalyzing(false);
            setAnalysisAbortController(null);
        }
    };

    const handleSynthesizeStyles = async () => {
        if (!activeProfileId || !activeProfile) return;
        const styleItems = activeProfile.library.filter(item => item.type === 'style');
        if (styleItems.length < 2) {
            showStatus("Es müssen mindestens 2 Stil-Elemente für eine Synthese vorhanden sein.", true);
            return;
        }
        
        setLoadingMessage('Analysiere & gruppiere Stile...'); // 1. Nachricht setzen
        setIsAnalyzing(true);                                 // 2. Overlay AN
        
        try {
            const payload = { style_items: styleItems };
            const response = await fetch(`${BACKEND_URL}/api/synthesize-styles`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) { const err = await response.json(); throw new Error(err.error || 'Stil-Synthese fehlgeschlagen'); }
            const result = await response.json();
            
            // --- NEUE LOGIK START ---
            setProfiles(prev => ({
                ...prev,
                [activeProfileId]: {
                    ...prev[activeProfileId],
                    styleClusters: result.style_clusters || []
                }
            }));
            // --- NEUE LOGIK ENDE ---

            setStyleViewMode('grouped');
            showStatus('Stile erfolgreich neu gruppiert!', false);
        } catch (error: any) {
            showStatus(error, true);
        } finally {
            setIsAnalyzing(false); // 3. Overlay AUS
        }
    };

    const handleToggleCluster = (clusterId: string) => {
        setOpenClusterIds(prev => ({
            ...prev,
            [clusterId]: !prev[clusterId]
        }));
    };

    const handleSynthesizeTechniques = async () => {
        if (!activeProfileId || !activeProfile) return;
        const techniqueItems = activeProfile.library.filter(item => item.type === 'technique');
        if (techniqueItems.length < 2) {
            showStatus("Es müssen mindestens 2 Technik-Elemente für eine Synthese vorhanden sein.", true);
            return;
        }
        
        setLoadingMessage('Analysiere & gruppiere Techniken...'); // 1. Nachricht setzen
        setIsAnalyzing(true);                                     // 2. Overlay AN
        
        try {
            const payload = { style_items: techniqueItems };
            const response = await fetch(`${BACKEND_URL}/api/synthesize-styles`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) { const err = await response.json(); throw new Error(err.error || 'Technik-Synthese fehlgeschlagen'); }
            const result = await response.json();

            // --- NEUE LOGIK START ---
            setProfiles(prev => ({
                ...prev,
                [activeProfileId]: {
                    ...prev[activeProfileId],
                    techniqueClusters: result.style_clusters || []
                }
            }));
            // --- NEUE LOGIK ENDE ---

            setTechniqueViewMode('grouped');
            showStatus('Techniken erfolgreich neu gruppiert!', false);
        } catch (error: any) {
            showStatus(error, true);
        } finally {
            setIsAnalyzing(false); // 3. Overlay AUS
        }
    };

    const handleToggleTechniqueCluster = (clusterId: string) => {
        setOpenTechniqueClusterIds(prev => ({
            ...prev,
            [clusterId]: !prev[clusterId]
        }));
    };

    const handleBeatFileAnalysis = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setIsAnalyzing(true); // Schaltet das Overlay AN

        const reader = new FileReader();
        reader.readAsDataURL(file);

        // Fehlerbehandlung für den Fall, dass die Datei nicht gelesen werden kann
        reader.onerror = () => {
            showStatus("Fehler beim Lesen der Datei.", true);
            setIsAnalyzing(false);
        };

        // Dieser Block wird erst ausgeführt, wenn die Datei fertig gelesen ist
        reader.onloadend = async () => {
            try {
                const base64Audio = (reader.result as string).split(',')[1];

                const response = await fetch(`${BACKEND_URL}/api/analyze-beat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        base64Audio: base64Audio,
                        mimeType: file.type 
                    }),
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.error || 'Beat-Analyse fehlgeschlagen');
                }

                const result = await response.json();

                setSongBPM(result.bpm || '');
                setSongKey(result.key || '');
                setSongBeatDescription(result.description || '');

                showStatus("Beat-Analyse erfolgreich abgeschlossen!", false);

            } catch (error: any) {
                showStatus(error, true);
            
            } finally {
                // WICHTIG: Schaltet das Overlay AUS, nachdem ALLES fertig ist
                setIsAnalyzing(false); 
                if (event.target) event.target.value = '';
            }
        };
    };

    const handleGenerateSongText = async () => {
        if (!songTopic.trim()) {
            showStatus("Bitte gib ein Thema für den Song ein.", true);
            return;
        }

        if (!activeProfileId) {
            showStatus("Kein aktives Profil ausgewählt.", true);
            return;
        }

        // Erstelle einen neuen AbortController für diese Analyse
        const abortController = new AbortController();
        setAnalysisAbortController(abortController);
        setIsAnalyzing(true);
        setLoadingMessage('Generiere Songtext basierend auf deiner Künstler-DNA...');
        
        // Warte kurz, damit der UI-Update sichtbar wird
        await new Promise(resolve => setTimeout(resolve, 100));

        try {
            // Hole styles und techniques direkt aus dem aktiven Profil
            const styles = activeProfile?.library.filter(i => i.type === 'style') ?? [];
            const techniques = activeProfile?.library.filter(i => i.type === 'technique') ?? [];
            
            // Sammle alle ausgewählten Zutaten
            const selectedStyles = songStyles.length > 0 
                ? styles.filter(s => songStyles.includes(s.id)).map(s => s.content)
                : [];
            const selectedTechniques = songTechniques.length > 0 
                ? techniques.filter(t => songTechniques.includes(t.id)).map(t => t.content)
                : [];

            // NEUE GUARD CLAUSE START
            const dnaResult = getUnifiedKnowledgeBase();
            if (!dnaResult.success) {
                showStatus(dnaResult.error!, true);
                return;
            }
            // NEUE GUARD CLAUSE ENDE

            // NEUE LOGIK ZUR PROMPT-ERSTELLUNG
            let structurePrompt = '';
            const performanceStyle = songPerformanceStyle === 'Hook (Gesungen)' ? 'gesungen' : songPerformanceStyle.toLowerCase();

            switch (songPartType) {
                case 'full_song':
                    structurePrompt = `Schreibe einen kompletten Songtext (zwei 16-zeilige Parts und eine 8-zeilige Hook). Der Performance-Stil der Parts ist ${performanceStyle}, die Hook ist immer gesungen.`;
                    break;
                case 'part_16':
                    structurePrompt = `Schreibe einen 16-zeiligen Part im Performance-Stil "${performanceStyle}".`;
                    break;
                case 'part_12':
                    structurePrompt = `Schreibe einen 12-zeiligen Part im Performance-Stil "${performanceStyle}".`;
                    break;
                case 'part_8':
                    structurePrompt = `Schreibe einen 8-zeiligen Part im Performance-Stil "${performanceStyle}".`;
                    break;
                case 'hook_8':
                    structurePrompt = `Schreibe eine 8-zeilige Hook (Refrain), die immer gesungen wird.`;
                    break;
                case 'bridge_8':
                    structurePrompt = `Schreibe eine 8-zeilige Bridge im Performance-Stil "${performanceStyle}".`;
                    break;
                case 'bridge_4':
                    structurePrompt = `Schreibe eine 4-zeilige Bridge im Performance-Stil "${performanceStyle}".`;
                    break;
                case 'custom':
                    structurePrompt = `Schreibe einen Text mit genau ${numLines} Zeilen im Performance-Stil "${performanceStyle}".`;
                    break;
                default:
                    structurePrompt = 'Schreibe einen Songtext.';
            }

            const finalUserPrompt = `
**AUFGABE:** ${structurePrompt}
**THEMA:** "${songTopic}"

**WICHTIGE ANWEISUNG:** Halte dich exakt an die in der AUFGABE definierte Struktur und Zeilenanzahl. Generiere NICHT mehr und nicht weniger als vorgegeben.
`;

            const payload = {
                mode: 'generation',
                knowledgeBase: dnaResult.knowledgeBase,
                userPrompt: finalUserPrompt, // <-- Hier den neuen, dynamischen Prompt verwenden
                additionalContext: {
                    style: selectedStyles,
                    technique: selectedTechniques,
                    beatDescription: songBeatDescription,
                    bpm: songBPM,
                    key: songKey,
                    performanceStyle: songPerformanceStyle
                }
            };

            const response = await fetch(`${BACKEND_URL}/api/generate-lyrics`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: abortController.signal
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Songtext-Generierung fehlgeschlagen');
            }

            const result = await response.json();
            
            // Zeige den generierten Songtext an
            setGeneratedSongText(result.generatedText || 'Kein Songtext generiert.');
            showStatus('Songtext erfolgreich generiert!', false);

        } catch (error: any) {
            if (error.name === 'AbortError') {
                showStatus('Songtext-Generierung wurde abgebrochen.', false);
            } else {
                showStatus(error, true);
            }
        } finally {
            setLoadingMessage('');
            setIsAnalyzing(false);
            setAnalysisAbortController(null);
        }
    };

    const handleSaveGeneratedSong = () => {
        if (!generatedSongText.trim() || !activeProfileId) return;

        // Erstelle ein neues Bibliothekselement für den generierten Text
        const newGeneratedLyric: LibraryItem = {
            id: `gen-lyric-${Date.now()}`,
            title: songTopic.trim() || 'Generierter Songtext',
            type: 'generated_lyric',
            content: generatedSongText
        };

        // Nutze die bestehende updateProfileLibrary-Funktion, um den State zu aktualisieren
        updateProfileLibrary(
            library => [...library, newGeneratedLyric],
            "Songtext erfolgreich in der Bibliothek gespeichert."
        );

        // NEU: Setze alle relevanten Eingabefelder zurück
        setSongTopic('');
        setSongStyles([]);
        setSongTechniques([]);
        setSongBeatDescription('');
        setSongBPM('');
        setSongKey('');
        setSongPartType('full_song');
        setSongPerformanceStyle('Gerappt');
        setGeneratedSongText(''); // Leert auch das Ausgabefeld
    };

    const handleSendTrainerMessage = async (text: string) => {
        if (!text.trim()) return;

        const userMessage = { text, isUser: true };
        const newMessages = [...trainerMessages, userMessage];
        setTrainerMessages(newMessages);
        setIsTrainerReplying(true);

        // Erstelle einen neuen AbortController für diese Analyse
        const abortController = new AbortController();
        setAnalysisAbortController(abortController);
        
        // Warte kurz, damit der UI-Update sichtbar wird
        await new Promise(resolve => setTimeout(resolve, 100));

        try {
            // NEUE GUARD CLAUSE START
            const dnaResult = getUnifiedKnowledgeBase();
            if (!dnaResult.success) {
                showStatus(dnaResult.error!, true);
                setIsTrainerReplying(false);
                return;
            }
            // NEUE GUARD CLAUSE ENDE

            const payload = {
                messages: newMessages,
                knowledgeBase: dnaResult.knowledgeBase
            };
            const response = await fetch(`${BACKEND_URL}/api/trainer-chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: abortController.signal
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Chat-Antwort fehlgeschlagen');
            }
            const result = await response.json();

            const aiMessage = { text: result.reply, isUser: false };
            setTrainerMessages(prev => [...prev, aiMessage]);

            // === NEUER BLOCK: Unterstützt alle drei Lern-Typen ===
            if (result.learning) {
                const learningItem = result.learning;

                // NEU: Unterscheide je nach Typ des Lern-Objekts
                switch (learningItem.type) {
                    case 'rule_category':
                        // Die bestehende Logik zum Speichern von Regeln
                        if (validateLearnedRule(learningItem)) {
                            updateProfileLibrary(
                                lib => [...lib, learningItem],
                                `Neue Regelkategorie '${learningItem.categoryTitle}' zur DNA hinzugefügt!`
                            );
                        }
                        break;
                    
                    case 'style':
                        // NEUE LOGIK: Erstelle und speichere ein Stil-Element
                        const newStyleItem: LibraryItem = {
                            id: `style-${Date.now()}`,
                            type: 'style',
                            content: learningItem.content,
                        };
                        updateProfileLibrary(lib => [...lib, newStyleItem], "Neuer Stil zur Bibliothek hinzugefügt.");
                        break;

                    case 'technique':
                        // NEUE LOGIK: Erstelle und speichere ein Technik-Element
                        const newTechniqueItem: LibraryItem = {
                            id: `technique-${Date.now()}`,
                            type: 'technique',
                            content: learningItem.content,
                        };
                        updateProfileLibrary(lib => [...lib, newTechniqueItem], "Neue Technik zur Bibliothek hinzugefügt.");
                        break;
                }
            } else {
                // Kein Lerninhalt beigebracht - nur Chat-Antwort
                console.log("Kein Lerninhalt beigebracht - nur Chat-Antwort");
            }
        } catch (error: any) {
            if (error.name === 'AbortError') {
                showStatus('Trainer-Chat wurde abgebrochen.', false);
            } else {
                showStatus(error, true);
            }
        } finally {
            setIsTrainerReplying(false);
            setAnalysisAbortController(null);
        }
    };

    // === VALIDIERUNGSFUNKTION FÜR GELERNTE REGELN ===
    const validateLearnedRule = (learningObject: any): boolean => {
        // Prüfe, ob das Objekt die grundlegende Struktur hat
        if (!learningObject || typeof learningObject !== 'object') {
            console.log("Kein gültiges Lern-Objekt");
            return false;
        }
        
        // Prüfe, ob es sich um eine rule_category handelt
        if (learningObject.type !== 'rule_category') {
            console.log("Kein rule_category Typ");
            return false;
        }
        
        // Prüfe, ob die erforderlichen Felder vorhanden sind
        if (!learningObject.categoryTitle || !learningObject.rules) {
            console.log("Fehlende erforderliche Felder");
            return false;
        }
        
        // Prüfe, ob es tatsächlich Regeln gibt
        if (!Array.isArray(learningObject.rules) || learningObject.rules.length === 0) {
            console.log("Keine Regeln im learningObject");
            return false;
        }
        
        // Prüfe jede einzelne Regel - nur grundlegende Struktur
        for (const rule of learningObject.rules) {
            if (!rule.title || !rule.definition) {
                console.log("Regel hat fehlende Felder:", rule);
                return false;
            }
            
            // Entferne die zu strenge Schlüsselwort-Prüfung
            // Nur prüfen, ob Titel und Definition nicht leer sind
            if (rule.title.trim().length === 0 || rule.definition.trim().length === 0) {
                console.log("Regel hat leere Felder:", rule);
                return false;
            }
        }
        
        console.log("Regel validiert - wird gespeichert");
        return true;
    };

    // === TRAINER CHAT FUNKTION ===

    // === RENDER-FUNKTIONEN ===
    const renderStartScreen = () => (
        <div className={`start-screen-container ${isStarting ? 'fading-out' : ''}`}>
            <div className="start-screen-content">
                <h1>Gallant's Lyric <span>Machine</span></h1>
                <p>Deine persönliche KI-Songwriting-Maschine.</p>
                <button className="start-button" onClick={(e) => {
                  const button = e.currentTarget;
                  const circle = document.createElement("span");
                  const diameter = Math.max(button.clientWidth, button.clientHeight);
                  const radius = diameter / 2;
                  circle.style.width = circle.style.height = `${diameter}px`;
                  // Ripple-Effekt startet immer vom Zentrum des Buttons
                  circle.style.left = `${radius}px`;
                  circle.style.top = `${radius}px`;
                  circle.style.transform = 'translate(-50%, -50%)';
                  circle.classList.add("ripple");
                  const ripple = button.getElementsByClassName("ripple")[0];
                  if (ripple) { ripple.remove(); }
                  button.appendChild(circle);
                  setIsStarting(true);
                  setTimeout(() => {
                      setCurrentView('transition');
                      setTimeout(() => {
                          setCurrentView('analyze');
                      }, 2400);
                  }, 500);
                }}>Start</button>
            </div>
        </div>
    );

    const renderTransitionScreen = () => (
        <div className="transition-screen">
            <div className="gallant-text">Made by Gallant</div>
        </div>
    );

    const renderModals = () => {
        // Handler, wenn der Nutzer die Aktion endgültig bestätigt
        const handleConfirmAction = () => {
            if (dontAskAgain && confirmationState.actionKey) {
                localStorage.setItem(`confirm_${confirmationState.actionKey}`, 'true');
            }
            confirmationState.onConfirm?.();
            setConfirmationState({ isOpen: false, message: '' }); // Modal schließen
            setDontAskAgain(false); // Reset für nächste Verwendung
        };

        return (
            <>
                {/* NEUES BESTÄTIGUNGS-MODAL */}
                {confirmationState.isOpen && (
                    <div className="modal-overlay">
                        <div className="modal-content v-stack v-stack--gap-md">
                            <h3>{confirmationState.message}</h3>
                            {confirmationState.details && <p className="panel-description">{confirmationState.details}</p>}
                            
                            <div className="modal-actions">
                                <label className="checkbox-label" style={{ marginRight: 'auto' }}>
                                    <input
                                        type="checkbox"
                                        checked={dontAskAgain}
                                        onChange={(e) => setDontAskAgain(e.target.checked)}
                                    />
                                    <span>Nicht mehr nachfragen</span>
                                </label>
                                <button className="secondary-action-button" onClick={() => setConfirmationState({ isOpen: false, message: '' })}>
                                    Abbrechen
                                </button>
                                <button className="danger-action-button" onClick={handleConfirmAction}>
                                    Bestätigen
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                <RhymeModal
                    isOpen={isRhymeModalOpen}
                    onClose={() => setIsRhymeModalOpen(false)}
                    targetWord={rhymeModalWord}
                    onSelectRhyme={handleSelectRhyme}
                    getKnowledgeBase={getUnifiedKnowledgeBase}
                />
                {isTitleModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-content v-stack v-stack--gap-md">
                        <SectionHeader
                            iconKey="TITLE_CONFIRM"
                            title="Titel bestätigen"
                            description="Die KI schlägt diesen Titel vor. Du kannst ihn hier anpassen."
                        />
                        <input type="text" value={editableTitle} onChange={e => setEditableTitle(e.target.value)} />
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsTitleModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleConfirmTitle}>Weiter</button>
                        </div>
                    </div>
                </div>
            )}
            {isLyricEditorModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-content wide">
                        <SectionHeader
                            iconKey="LYRIC_EDIT"
                            title="Transkription bearbeiten"
                            description="Korrigiere hier den transkribierten Text, bevor die endgültige Tiefenanalyse gestartet wird."
                        />
                        <textarea value={editableLyrics} onChange={e => setEditableLyrics(e.target.value)} className="lyrics-editor"/>
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsLyricEditorModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleConfirmEditedLyrics}>Analyse starten</button>
                        </div>
                    </div>
                </div>
            )}
            {isCreateProfileModalOpen && (
                <div className="modal-overlay">
                    {/* Hier die v-stack Klassen für den automatischen Abstand hinzufügen */}
                    <div className="modal-content v-stack v-stack--gap-md">
                        <SectionHeader
                            iconKey="PROFILE_CREATE"
                            title="Neues Profil erstellen"
                            description="Erstelle ein neues Profil für deine Künstler-DNA."
                        />
                        <input type="text" value={newProfileName} onChange={e => setNewProfileName(e.target.value)} placeholder="Name des neuen Profils"/>
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsCreateProfileModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleCreateProfile}>Erstellen</button>
                        </div>
                    </div>
                </div>
            )}
            {isEditModalOpen && editingItem && (
                 <div className="modal-overlay">
                    <div className="modal-content wide">
                        <SectionHeader
                            iconKey="ITEM_EDIT"
                            title={`"${editingItem.title || editingItem.type}" bearbeiten`}
                            description="Bearbeite den Inhalt dieses Elements."
                        />
                        <textarea value={editedContent} onChange={(e) => setEditedContent(e.target.value)} className="lyrics-editor"/>
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsEditModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleSaveEdit}>Speichern</button>
                        </div>
                    </div>
                </div>
            )}
            {isTransferProfileModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-content v-stack v-stack--gap-md">
                        <SectionHeader
                            iconKey="PROFILE_TRANSFER"
                            title="Profildaten übertragen"
                            description="Übertrage alle gespeicherten Daten von einem Profil auf ein anderes."
                        />
                        
                        <div className="form-group">
                            <label>Von Profil (Quelle):</label>
                            <select 
                                value={transferSourceProfileId} 
                                onChange={e => setTransferSourceProfileId(e.target.value)}
                            >
                                <option value="">Quellprofil auswählen...</option>
                                {Object.values(profiles).map(p => (
                                    <option key={p.id} value={p.id}>
                                        {p.name} ({p.library.length} Einträge)
                                    </option>
                                ))}
                            </select>
                        </div>
                        
                        <div className="form-group">
                            <label>Auf Profil (Ziel):</label>
                            <select 
                                value={transferTargetProfileId} 
                                onChange={e => setTransferTargetProfileId(e.target.value)}
                            >
                                <option value="">Zielprofil auswählen...</option>
                                {Object.values(profiles).map(p => (
                                    <option key={p.id} value={p.id}>
                                        {p.name} ({p.library.length} Einträge)
                                    </option>
                                ))}
                            </select>
                        </div>
                        
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => {
                                setIsTransferProfileModalOpen(false);
                                setTransferSourceProfileId('');
                                setTransferTargetProfileId('');
                            }}>Abbrechen</button>
                            <button 
                                className="main-action-button" 
                                onClick={handleTransferProfile}
                                disabled={!transferSourceProfileId || !transferTargetProfileId}
                            >
                                Daten übertragen
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
        );
    };

const renderAppHeader = () => (
    <header className="app-header">
        <h1>Gallant's Lyric <span>Machine</span></h1>
        {activeProfile && (
            <div className="profile-selector header-profile-selector">
                <span>Profil:</span>
                <select value={activeProfileId ?? ''} onChange={e => setActiveProfileId(e.target.value)}>
                    {Object.values(profiles).map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
                <button
                    className="add-profile-button"
                    title="Neues Profil erstellen"
                    onClick={() => setIsCreateProfileModalOpen(true)}
                >
                    {UI_ICONS.ADD}
                </button>
                <button
                    className="secondary-action-button"
                    title="Profildaten übertragen"
                    onClick={() => setIsTransferProfileModalOpen(true)}
                >
                    Daten übertragen
                </button>
            </div>
        )}
    </header>
);

    const renderSideMenu = () => {
    // 1. Stelle sicher, dass dein Array so aufgebaut ist:
    const menuItems: { key: View, label: string, iconKey: string }[] = [
        { key: 'analyze', label: 'Song analysieren', iconKey: 'ANALYZE' },
        { key: 'trainer', label: 'KI-Trainer', iconKey: 'TRAINER' },
        { key: 'rhyme_machine', label: 'Rhyme Machine', iconKey: 'RHYME' },
        { key: 'write_song', label: 'Der Ghostwriter', iconKey: 'WRITE' },
        { key: 'manage_library', label: 'Bibliothek verwalten', iconKey: 'LIBRARY' },
        { key: 'kuenstler_dna', label: 'Künstler DNA', iconKey: 'DNA' }
        ];

        return (
            <aside className="side-menu">
                <div className="menu-header">Menü</div>
                <nav>
                    {menuItems.map(item => (
                        <button key={item.key} className={`menu-button ${currentView === item.key ? 'active' : ''}`} onClick={() => setCurrentView(item.key)}>
                        <span className="menu-icon">
                            {/* 2. Stelle sicher, dass das Icon hier nachgeschlagen wird: */}
                            {MAIN_ICONS[item.iconKey] || MAIN_ICONS.DEFAULT}
                        </span>
                            <span>{item.label}</span>
                        </button>
                    ))}
                </nav>
            </aside>
        );
    };

const renderTrainerView = () => (
            <Panel title="KI-Trainer" iconKey="TRAINER" description="Führe ein Gespräch mit der KI, um ihr neue Regeln, Stile, oder Techniken beizubringen. Jede gelernte Lektion wird permanent in der Bibliothek des aktiven Profils gespeichert.">
        <div className="chat-section">
            <div className="chat-container v-stack v-stack--gap-sm">
                <div className="chat-messages">
                    {trainerMessages.map((msg, index) => (
                        <div key={index} className={`chat-bubble ${msg.isUser ? 'user' : 'ai'}`}>
                            <div className="author">
                                {msg.isUser ? (activeProfile?.name || 'Benutzer') : 'KI-Trainer'}
                            </div>
                            {msg.text}
                        </div>
                    ))}
                    {isTrainerReplying && (
                        <div className="chat-bubble ai">
                            <div className="author">KI-Trainer</div>
                            <div className="typing-indicator">
                                <div className="typing-dot"></div>
                                <div className="typing-dot"></div>
                                <div className="typing-dot"></div>
                            </div>
                        </div>
                    )}
                </div>
                <div className="chat-input-section">
                    {isTrainerReplying ? (
                        <>
                            <div className="loading-bar">
                                <div className="loading-progress"></div>
                            </div>
                            <button className="danger-action-button" onClick={handleAbortAnalysis}>
                                Chat abbrechen
                            </button>
                        </>
                    ) : (
                        <ChatInput onSend={handleSendTrainerMessage} isReplying={isTrainerReplying} />
                    )}
                </div>
            </div>
        </div>
    </Panel>
);

const renderAnalyzeView = () => (
    <Panel title="Lyric Analyzer" iconKey="ANALYZE" description="Analysiere eine oder mehrere Audiodateien, um Songs inklusive Stil, Technik und Rhythmus automatisch in deine Bibliothek aufzunehmen.">
        <div className="v-stack v-stack--gap-lg">
            <SectionHeader
                iconKey="AUDIO_ANALYSIS"
                title="Audio-Analyse"
                description="Lade Audio-Dateien hoch, um deine Bibliothek zu erweitern."
            />

            <div className="main-action-container">
                {isAnalyzing ? (
                    <div className="loading-container">
                        <div className="typing-indicator">
                            <div className="typing-dot"></div>
                            <div className="typing-dot"></div>
                            <div className="typing-dot"></div>
                        </div>
                        <p className="loading-text">Analysiere Songs...</p>
                        <button className="danger-action-button" onClick={handleAbortAnalysis}>
                            Analyse abbrechen
                        </button>
                    </div>
                ) : (
                    <div className="button-container">
                        <button className="main-action-button" onClick={() => fileInputRef.current?.click()}>
                            Songs analysieren (Audio-Dateien)
                        </button>
                        <button 
                            className={`record-button ${isRecording ? 'recording' : ''}`} 
                            title={isRecording ? "Aufnahme beenden" : "Song direkt aufnehmen"}
                            onClick={handleToggleRecording}
                        >
                            <span className="record-dot"></span>
                        </button>
                    </div>
                )}
            </div>
            
            <input type="file" ref={fileInputRef} onChange={handleFileUpload} style={{ display: 'none' }} accept="audio/*" />
            
            <div className="divider">ODER</div>
            
            <div className="text-input-section v-stack v-stack--gap-md">
                <SectionHeader
                    iconKey="TEXT_INPUT"
                    title="Manuelle Texteingabe"
                    description="Füge hier Text manuell ein, um ihn zu analysieren."
                />
                
                <div className="form-group">
                    <label>Songtitel (optional)</label>
                    <MemoizedInput
                        placeholder="Songtitel (optional)"
                        value={manualTitle}
                        onChange={e => setManualTitle(e.target.value)}
                        className="text-input"
                    />
                </div>

                <div className="form-group">
                    <label>Songtext</label>
                    <MemoizedTextarea
                        placeholder="Songtext hier einfügen..."
                        value={manualLyrics}
                        onChange={e => setManualLyrics(e.target.value)}
                        className="lyrics-editor" /* <-- Korrekte Klasse wieder zugewiesen */
                    />
                </div>
                
                <div className="button-container">
                    {isAnalyzing ? (
                        <div className="loading-container">
                            <div className="typing-indicator">
                                <div className="typing-dot"></div>
                                <div className="typing-dot"></div>
                                <div className="typing-dot"></div>
                            </div>
                            <p className="loading-text">Analysiere Text...</p>
                            <button className="danger-action-button" onClick={handleAbortAnalysis}>
                                Analyse abbrechen
                            </button>
                        </div>
                    ) : (
                        <button className="main-action-button" onClick={handleAnalyzeManualText}>
                            Text analysieren
                        </button>
                    )}
                </div>
            </div>
        </div>
    </Panel>
);

const renderLibraryView = () => {
    // Vereinfachte Liste der Tabs für eine saubere Anzeige
    const tabs: { key: LibraryTab, label: string }[] = [
        { key: 'learned_lyrics', label: 'Gelernte Lyrics' },
        { key: 'learned_rules', label: 'Gelernte Regeln' },
        { key: 'rhyme_lessons', label: 'Gelernte Reime' },
        { key: 'generated_lyrics', label: 'Generierte Lyrics' },
        { key: 'style', label: 'Character / Stil' },
        { key: 'technique', label: 'Technik' },
        { key: 'emphasis', label: 'Betonung' },
        { key: 'rhyme_flow', label: 'Reimfluss' },
    ];

    const getTabCount = (tabKey: LibraryTab) => {
        if (!activeProfile) return 0;
        const lib = activeProfile.library;
        switch(tabKey) {
            case 'learned_lyrics': return lib.filter(i => i.type === 'lyric').length;
            case 'learned_rules': return lib.filter(i => i.type === 'rule_category').length;
            case 'rhyme_lessons': return lib.filter(i => i.type === 'rhyme_lesson_group').length;
            case 'generated_lyrics': return lib.filter(i => i.type === 'generated_lyric').length;
            case 'style': return lib.filter(i => i.type === 'style').length;
            case 'technique': return lib.filter(i => i.type === 'technique').length;
            case 'emphasis': return lib.filter(i => i.type === 'emphasis').length;
            case 'rhyme_flow': return lib.filter(i => i.type === 'rhyme_flow').length;
            default: return 0;
        }
    };

    const filterItemsForTab = (tabKey: LibraryTab) => {
        if (!activeProfile) return [];
        const lib = activeProfile.library;
        switch(tabKey) {
            case 'learned_lyrics': return lib.filter(i => i.type === 'lyric');
            case 'learned_rules': return lib.filter(i => i.type === 'rule_category');
            case 'rhyme_lessons': return lib.filter(i => i.type === 'rhyme_lesson_group');
            case 'generated_lyrics': return lib.filter(i => i.type === 'generated_lyric');
            case 'style': return lib.filter(i => i.type === 'style');
            case 'technique': return lib.filter(i => i.type === 'technique');
            case 'emphasis': return lib.filter(i => i.type === 'emphasis');
            case 'rhyme_flow': return lib.filter(i => i.type === 'rhyme_flow');
            default: return [];
        }
    };

    const filteredItems = filterItemsForTab(libraryTab);

    // NEUE, KORREKTE SORTIERLOGIK
    filteredItems.sort((a, b) => {
        // Für "lyric"-Einträge, sortiere nach dem Titel
        if (a.type === 'lyric' && b.type === 'lyric') {
            return (a.title || '').localeCompare(b.title || '');
        }
        // Für alle anderen Einträge, sortiere nach dem Inhalt
        // Dies betrifft Character, Technik, Betonung und Reimfluss
        else if (a.type !== 'lyric' && b.type !== 'lyric') {
            // Für Betonung & Reimfluss bauen wir den dynamischen Titel
            if (a.type === 'emphasis' || a.type === 'rhyme_flow') {
                const sourceSongA = activeProfile?.library.find(s => s.id === a.sourceLyricId);
                const sourceSongB = activeProfile?.library.find(s => s.id === b.sourceLyricId);
                const titleA = sourceSongA ? sourceSongA.title || '' : '';
                const titleB = sourceSongB ? sourceSongB.title || '' : '';
                return titleA.localeCompare(titleB);
            }
            // Für Character & Technik, nutze den Inhalt
            return (a.content || '').localeCompare(b.content || '');
        }
        // Fallback, falls Typen gemischt sind (sollte nicht passieren)
        return 0;
    });

    return (
        <>
             {/* NEUES MODAL FÜR REIM-BEARBEITUNG */}
        {isSubRhymeEditModalOpen && editingSubRhyme && (
            <div className="modal-overlay">
                <div className="modal-content">
                    <SectionHeader
                        iconKey="RHYME_EDIT"
                        title="Reim bearbeiten"
                        description="Bearbeite den Inhalt dieses Reims."
                    />
                    <input type="text" value={editedSubRhymeContent} onChange={e => setEditedSubRhymeContent(e.target.value)} autoFocus/>
                    <div className="modal-actions">
                        <button className="secondary-action-button" onClick={() => setIsSubRhymeEditModalOpen(false)}>Abbrechen</button>
                        <button className="main-action-button" onClick={handleSaveSubRhymeEdit}>Speichern</button>
                    </div>
                </div>
            </div>
        )}

            {isRuleEditModalOpen && editingRule && (
                <div className="modal-overlay">
                    <div className="modal-content wide">
                        <SectionHeader
                            iconKey="RULE_EDIT"
                            title={`"${editingRule.rule.title}" bearbeiten`}
                            description="Bearbeite die Definition dieser Regel."
                        />
                        <textarea
                            value={editedRuleContent}
                            onChange={(e) => setEditedRuleContent(e.target.value)}
                            className="lyrics-editor"
                        />
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsRuleEditModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleSaveRuleEdit}>Speichern</button>
                        </div>
                    </div>
                </div>
            )}
            <Panel title="Bibliothek verwalten" iconKey="LIBRARY" description="Hier werden alle gelernten Elemente deiner Künstler-DNA gespeichert und verwaltet.">
                <div className="panel-content v-stack v-stack--gap-md">
                    <div className="library-header">
                        <div className="library-tabs">
                            {tabs.map(tab => (
                                <button key={tab.key} className={`tab-button ${libraryTab === tab.key ? 'active' : ''}`} onClick={() => setLibraryTab(tab.key)}>
                                    {tab.label} ({getTabCount(tab.key)})
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* NEU: Button zum Umschalten der Ansicht */}
                    {['emphasis', 'rhyme_flow'].includes(libraryTab) && (
                        <div className="library-controls">
                            <button 
                                className="secondary-action-button" 
                                onClick={() => setDetailViewMode(prev => prev === 'summary' ? 'full' : 'summary')}
                            >
                                {detailViewMode === 'summary' ? 'Zum Volltext wechseln' : 'Zur Zusammenfassung wechseln'}
                            </button>
                        </div>
                    )}

                    <div className="library-content">
                    {filteredItems.length === 0 && <p>Diese Sektion ist leer.</p>}

                    {libraryTab === 'learned_rules' && filteredItems.map(item => {
                        const category = item as RuleCategory;
                        return (
                            <div key={category.id} className="accordion-item">
                                <div className="accordion-header" onClick={() => setOpenRuleCategoryId(openRuleCategoryId === category.id ? null : category.id)}>
                                    <span onDoubleClick={() => setIsEditingCategoryTitle({ id: category.id, title: category.categoryTitle })}>{category.categoryTitle}</span>
                                    <div className={`accordion-icon ${openRuleCategoryId === category.id ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                                </div>
                                {openRuleCategoryId === category.id && (
                                    <div className="accordion-content">
                                        {/* HIER die Sortierung hinzufügen: */}
                                        {[...(category.rules || [])]
                                            .sort((a, b) => a.title.localeCompare(b.title))
                                            .map(rule => (
                                                <div key={rule.id} className="simple-library-item rule-item">
                                                    <strong onDoubleClick={() => setIsEditingRuleTitle({ categoryId: category.id, ruleId: rule.id, title: rule.title })}>{rule.title}</strong>
                                                    <p onDoubleClick={() => setIsEditingRuleDefinition({ categoryId: category.id, ruleId: rule.id, definition: rule.definition })}>{rule.definition}</p>
                                                    <div className="item-actions">
                                                        <button className="action-button" onClick={() => handleStartRuleEdit(category.id, rule)}>Bearbeiten</button>
                                                        <button 
                                                            className="action-button delete" 
                                                            onClick={() => requestConfirmation({
                                                                message: 'Regel wirklich löschen?',
                                                                details: 'Dadurch wird diese Regel endgültig aus der Kategorie entfernt.',
                                                                actionKey: 'delete_rule',
                                                                onConfirm: () => handleDeleteRule(category.id, rule.id)
                                                            })}
                                                        >
                                                            Löschen
                                                        </button>
                                                    </div>
                                                </div>
                                            ))
                                        }
                                        <div className="item-actions top">
                                            <button 
                                                className="action-button delete" 
                                                onClick={() => requestConfirmation({
                                                    message: 'Ganze Kategorie wirklich löschen?',
                                                    details: `Dadurch wird die Kategorie "${category.categoryTitle}" mit allen ${category.rules.length} Regel(n) endgültig gelöscht.`,
                                                    actionKey: 'delete_rule_category',
                                                    onConfirm: () => handleDeleteRuleCategory(category.id)
                                                })}
                                            >
                                                Kategorie löschen
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}

                    {(libraryTab === 'learned_lyrics' || libraryTab === 'generated_lyrics') && filteredItems.map(item => {
                         const lyric = item as LibraryItem;
                         return (
                            <div key={lyric.id} className="accordion-item">
                                <div className="accordion-header" onClick={() => setOpenLyricId(openLyricId === lyric.id ? null : lyric.id)}>
                                    <span>{lyric.title || 'Unbenannter Song'}</span>
                                    <div className={`accordion-icon ${openLyricId === lyric.id ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                                </div>
                                {openLyricId === lyric.id && (
                                    <div className="accordion-content">
                                        <div className="item-actions top">
                                            <button className="action-button" onClick={() => handleStartEditing(lyric)}>Text bearbeiten</button>
                                            <button 
                                                className="action-button delete" 
                                                onClick={() => requestConfirmation({
                                                    message: 'Eintrag wirklich löschen?',
                                                    details: 'Dadurch wird dieser Eintrag und alle damit verbundenen Elemente endgültig aus deiner Bibliothek entfernt.',
                                                    actionKey: 'delete_library_item',
                                                    onConfirm: () => handleDeleteItem(lyric.id)
                                                })}
                                            >
                                                Song löschen
                                            </button>
                                        </div>
                                        <h4>Originaltext</h4><pre>{lyric.content}</pre>
                                    </div>
                                )}
                            </div>
                         );
                    })}

                    {libraryTab === 'rhyme_lessons' && (() => {
    const lessonGroups = filteredItems as RhymeLessonGroup[];

    const groupedByVowelSequence = lessonGroups.reduce((acc, group) => {
        const key = group.vowelSequence || 'Unbekannt';
        if (!acc[key]) { acc[key] = []; }
        acc[key].push(group);
        return acc;
    }, {} as Record<string, RhymeLessonGroup[]>);

    return Object.entries(groupedByVowelSequence).map(([sequence, groupsInSequence]) => (
        <div key={sequence} className="accordion-item">
            <div className="accordion-header level-1" onClick={() => setOpenRuleCategoryId(openRuleCategoryId === sequence ? null : sequence)}>
                <span>Vokalfolge: {sequence}</span>
                <div className={`accordion-icon ${openRuleCategoryId === sequence ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
            </div>
            {openRuleCategoryId === sequence && (
                <div className="accordion-content">
                    {groupsInSequence.map(group => (
                        <div key={group.id} className="accordion-item nested">
                            <div className="accordion-header level-2" onClick={() => setOpenLyricId(openLyricId === group.id ? null : group.id)}>
                                <span>Ausgangswort: <strong>{group.targetWord}</strong> ({group.rhymes?.length || 0} Reime)</span>
                                <div className={`accordion-icon ${openLyricId === group.id ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                            </div>
                            {openLyricId === group.id && (
                                <div className="accordion-content">
                                    {/* HIER die Sortierung hinzufügen: */}
                                    {[...(group.rhymes || [])]
                                        .sort((a, b) => a.rhymingWord.localeCompare(b.rhymingWord))
                                        .map((rhyme) => (
                                            <div key={rhyme.id} className="simple-library-item rule-item">
                                                <p>{rhyme.rhymingWord}</p>
                                                <div className="item-actions">
                                                    <button className="action-button" onClick={() => handleStartSubRhymeEdit(group.id, rhyme)}>Bearbeiten</button>
                                                    <button 
                                                        className="action-button delete" 
                                                        onClick={() => requestConfirmation({
                                                            message: 'Reim wirklich löschen?',
                                                            details: 'Dadurch wird dieser Reim endgültig aus der Reimgruppe entfernt.',
                                                            actionKey: 'delete_sub_rhyme',
                                                            onConfirm: () => handleDeleteSubRhyme(group.id, rhyme.id)
                                                        })}
                                                    >
                                                        Löschen
                                                    </button>
                                                </div>
                                            </div>
                                        ))
                                    }
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    ));
})()}

                    {libraryTab === 'style' && (
                        <>
                            {/* Lese Daten direkt aus dem Profil */}
                            {(() => {
                                const styleClusters = activeProfile?.styleClusters;
                                return (
                                    <>
                                        <div className="library-controls">
                                            {/* --- INTELLIGENTER UMSCHALT-BUTTON --- */}
                                            {styleClusters && styleClusters.length > 0 ? (
                                                // Wenn Cluster-Daten vorhanden sind, zeige den Umschalt-Button an
                                                <button 
                                                    className="secondary-action-button" 
                                                    onClick={() => setStyleViewMode(prev => prev === 'single' ? 'grouped' : 'single')}
                                                >
                                                    {styleViewMode === 'single' ? 'Zur Gruppenansicht wechseln' : 'Zur Einzelansicht wechseln'}
                                                </button>
                                            ) : (
                                                // Andernfalls zeige den ursprünglichen Analyse-Button
                                                <button className="secondary-action-button" onClick={handleSynthesizeStyles} disabled={isSynthesizing}>
                                                    {isSynthesizing ? 'Optimiere...' : 'Stile optimieren & gruppieren'}
                                                </button>
                                            )}
                                        </div>

                                        {/* --- BEDINGTE ANZEIGE BASIEREND AUF DEM MODUS --- */}
                                        {(styleViewMode === 'grouped' && styleClusters) ? (
                                            <div className="style-cluster-container">
                                                {/* 1. Sortiere die Cluster-Gruppen alphabetisch nach ihrem Titel */}
                                                {[...styleClusters]
                                                    .sort((a, b) => a.cluster_title.localeCompare(b.cluster_title))
                                                    .map(cluster => (
                                                        <div key={cluster.cluster_id} className="accordion-item">
                                                            <div className="accordion-header" onClick={() => handleToggleCluster(cluster.cluster_id)}>
                                                                <span>{cluster.cluster_title} <span className="category-tag">{cluster.category}</span></span>
                                                                <div className={`accordion-icon ${openClusterIds[cluster.cluster_id] ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                                                            </div>
                                                            {openClusterIds[cluster.cluster_id] && (
                                                                <div className="accordion-content open">
                                                                    <div className="item-list-container">
                                                                        {/* 2. Sortiere die Einträge INNERHALB des Clusters alphabetisch */}
                                                                        {[...cluster.facets]
                                                                            .sort((a, b) => a.content.localeCompare(b.content))
                                                                            // HIER wird der 'index' hinzugefügt
                                                                            .map((facet: { id: string, content: string }, index) => { 
                                                                                const originalItem = activeProfile?.library.find(i => i.id === facet.id);
                                                                                const sourceSong = originalItem?.sourceLyricId ? activeProfile?.library.find(s => s.id === originalItem.sourceLyricId) : null;
                                                                                return (
                                                                                    // HIER wird der neue, kombinierte key verwendet
                                                                                    <div key={`${cluster.cluster_id}-${facet.id}-${index}`} className="simple-library-item facet-item">
                                                                                        <p>{facet.content}</p>
                                                                                        <div className="facet-meta">
                                                                                            {sourceSong && <span className="source-song-tag">(aus "{sourceSong.title}")</span>}
                                                                                            {/* HIER WURDE DER ONCLICK-HANDLER GEÄNDERT */}
                                                                                            <button 
                                                                                                className="action-button delete-facet" 
                                                                                                title="Diesen Eintrag löschen"
                                                                                                onClick={() => requestConfirmation({
                                                                                                    message: 'Eintrag wirklich löschen?',
                                                                                                    details: 'Dadurch wird dieser Eintrag endgültig aus deiner Bibliothek entfernt.',
                                                                                                    actionKey: 'delete_library_item_facet',
                                                                                                    onConfirm: () => handleDeleteItem(facet.id)
                                                                                                })}
                                                                                            >
                                                                                                {UI_ICONS.CLOSE}
                                                                                            </button>
                                                                                        </div>
                                                                                    </div>
                                                                                );
                                                                            })
                                                                        }
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))
                                                }
                                            </div>
                                        ) : (
                                            // NEUE AKKORDEON-ANSICHT für groupedItems
                                            groupedItems && groupedItems.map(group => (
                                                <div key={group.lyric.id} className="accordion-item">
                                                    <div className="accordion-header" onClick={() => setOpenLyricId(openLyricId === group.lyric.id ? null : group.lyric.id)}>
                                                        <span>Aus "{group.lyric.title || 'Unbenannter Song'}"</span>
                                                        <div className={`accordion-icon ${openLyricId === group.lyric.id ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                                                    </div>
                                                    {openLyricId === group.lyric.id && (
                                                        <div className="accordion-content open">
                                                            <div className="item-list-container">
                                                                {group.items.map(item => (
                                                                    <div key={item.id} className="simple-library-item">
                                                                        <p>{item.content}</p>
                                                                        <div className="item-actions">
                                                                            <button className="action-button" onClick={() => handleStartEditing(item as LibraryItem)}>Bearbeiten</button>
                                                                            {/* HIER WURDE DER ONCLICK-HANDLER GEÄNDERT */}
                                                                            <button 
                                                                                className="action-button delete" 
                                                                                onClick={() => requestConfirmation({
                                                                                    message: 'Eintrag wirklich löschen?',
                                                                                    details: 'Dadurch wird dieser Eintrag endgültig aus deiner Bibliothek entfernt.',
                                                                                    actionKey: 'delete_library_item',
                                                                                    onConfirm: () => handleDeleteItem(item.id)
                                                                                })}
                                                                            >
                                                                                Löschen
                                                                            </button>
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            ))
                                        )}
                                    </>
                                );
                            })()}
                        </>
                    )}

                    {libraryTab === 'technique' && (
                        <>
                            {/* Lese Daten direkt aus dem Profil */}
                            {(() => {
                                const techniqueClusters = activeProfile?.techniqueClusters;
                                return (
                                    <>
                                        <div className="library-controls">
                                            {/* --- INTELLIGENTER UMSCHALT-BUTTON --- */}
                                            {techniqueClusters && techniqueClusters.length > 0 ? (
                                                // Wenn Cluster-Daten vorhanden sind, zeige den Umschalt-Button an
                                                <button 
                                                    className="secondary-action-button" 
                                                    onClick={() => setTechniqueViewMode(prev => prev === 'single' ? 'grouped' : 'single')}
                                                >
                                                    {techniqueViewMode === 'single' ? 'Zur Gruppenansicht wechseln' : 'Zur Einzelansicht wechseln'}
                                                </button>
                                            ) : (
                                                // Andernfalls zeige den ursprünglichen Analyse-Button
                                                <button className="secondary-action-button" onClick={handleSynthesizeTechniques} disabled={isSynthesizingTechniques}>
                                                    {isSynthesizingTechniques ? 'Optimiere...' : 'Techniken optimieren & gruppieren'}
                                                </button>
                                            )}
                                        </div>

                                        {/* --- BEDINGTE ANZEIGE BASIEREND AUF DEM MODUS --- */}
                                        {(techniqueViewMode === 'grouped' && techniqueClusters) ? (
                                            <div className="technique-cluster-container">
                                                {/* 1. Sortiere die Cluster-Gruppen alphabetisch nach ihrem Titel */}
                                                {[...techniqueClusters]
                                                    .sort((a, b) => a.cluster_title.localeCompare(b.cluster_title))
                                                    .map(cluster => (
                                                        <div key={cluster.cluster_id} className="accordion-item">
                                                            <div className="accordion-header" onClick={() => handleToggleTechniqueCluster(cluster.cluster_id)}>
                                                                <span>{cluster.cluster_title} <span className="category-tag">{cluster.category}</span></span>
                                                                <div className={`accordion-icon ${openTechniqueClusterIds[cluster.cluster_id] ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                                                            </div>
                                                            {openTechniqueClusterIds[cluster.cluster_id] && (
                                                                <div className="accordion-content open">
                                                                    <div className="item-list-container">
                                                                        {/* 2. Sortiere die Einträge INNERHALB des Clusters alphabetisch */}
                                                                        {[...cluster.facets]
                                                                            .sort((a, b) => a.content.localeCompare(b.content))
                                                                            // HIER wird der 'index' hinzugefügt
                                                                            .map((facet: { id: string, content: string }, index) => { 
                                                                                const originalItem = activeProfile?.library.find(i => i.id === facet.id);
                                                                                const sourceSong = originalItem?.sourceLyricId ? activeProfile?.library.find(s => s.id === originalItem.sourceLyricId) : null;
                                                                                return (
                                                                                    // HIER wird der neue, kombinierte key verwendet
                                                                                    <div key={`${cluster.cluster_id}-${facet.id}-${index}`} className="simple-library-item facet-item">
                                                                                        <p>{facet.content}</p>
                                                                                        <div className="facet-meta">
                                                                                            {sourceSong && <span className="source-song-tag">(aus "{sourceSong.title}")</span>}
                                                                                            {/* HIER WURDE DER ONCLICK-HANDLER GEÄNDERT */}
                                                                                            <button 
                                                                                                className="action-button delete-facet" 
                                                                                                title="Diesen Eintrag löschen"
                                                                                                onClick={() => requestConfirmation({
                                                                                                    message: 'Eintrag wirklich löschen?',
                                                                                                    details: 'Dadurch wird dieser Eintrag endgültig aus deiner Bibliothek entfernt.',
                                                                                                    actionKey: 'delete_library_item_facet',
                                                                                                    onConfirm: () => handleDeleteItem(facet.id)
                                                                                                })}
                                                                                            >
                                                                                                {UI_ICONS.CLOSE}
                                                                                            </button>
                                                                                        </div>
                                                                                    </div>
                                                                                );
                                                                            })
                                                                        }
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))
                                                }
                                            </div>
                                        ) : (
                                            // NEUE AKKORDEON-ANSICHT für groupedItems
                                            groupedItems && groupedItems.map(group => (
                                                <div key={group.lyric.id} className="accordion-item">
                                                    <div className="accordion-header" onClick={() => setOpenLyricId(openLyricId === group.lyric.id ? null : group.lyric.id)}>
                                                        <span>Aus "{group.lyric.title || 'Unbenannter Song'}"</span>
                                                        <div className={`accordion-icon ${openLyricId === group.lyric.id ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                                                    </div>
                                                    {openLyricId === group.lyric.id && (
                                                        <div className="accordion-content open">
                                                            <div className="item-list-container">
                                                                {group.items.map(item => (
                                                                    <div key={item.id} className="simple-library-item">
                                                                        <p>{item.content}</p>
                                                                        <div className="item-actions">
                                                                            <button className="action-button" onClick={() => handleStartEditing(item as LibraryItem)}>Bearbeiten</button>
                                                                            {/* HIER WURDE DER ONCLICK-HANDLER GEÄNDERT */}
                                                                            <button 
                                                                                className="action-button delete" 
                                                                                onClick={() => requestConfirmation({
                                                                                    message: 'Eintrag wirklich löschen?',
                                                                                    details: 'Dadurch wird dieser Eintrag endgültig aus deiner Bibliothek entfernt.',
                                                                                    actionKey: 'delete_library_item',
                                                                                    onConfirm: () => handleDeleteItem(item.id)
                                                                                })}
                                                                            >
                                                                                Löschen
                                                                            </button>
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            ))
                                        )}
                                    </>
                                );
                            })()}
                        </>
                    )}

                    {libraryTab === 'emphasis' && filteredItems.map(item => {
                        const sourceSong = activeProfile?.library.find(song => song.id === item.sourceLyricId);
                        const displayTitle = sourceSong
                            ? `Betonung aus "${sourceSong.title || 'Unbenannter Song'}"`
                            : 'Unbekanntes Betonungsmuster';

                        return (
                            <div key={item.id} className="accordion-item">
                                <div className="accordion-header" onClick={() => setOpenLyricId(openLyricId === item.id ? null : item.id)}>
                                    <span>{displayTitle}</span>
                                    <div className={`accordion-icon ${openLyricId === item.id ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                                </div>
                                {openLyricId === item.id && (
                                    <div className="accordion-content">
                                        <div className="item-actions top">
                                            {/* NEUER BUTTON FÜR OPTIMIERUNG */}
                                            {!item.structuredData && (
                                                 <button 
                                                    className="secondary-action-button" 
                                                    onClick={() => handleOptimizeForDna(item.id)}
                                                    disabled={optimizingItemId === item.id}
                                                >
                                                    {optimizingItemId === item.id ? 'Optimiere...' : 'Für DNA optimieren'}
                                                </button>
                                            )}
                                            <button className="action-button" onClick={() => handleStartEditing(item)}>Bearbeiten</button>
                                            <button 
                                                className="action-button delete" 
                                                onClick={() => requestConfirmation({
                                                    message: 'Betonungsmuster wirklich löschen?',
                                                    details: 'Dadurch wird dieses Betonungsmuster und alle damit verbundenen Elemente endgültig aus deiner Bibliothek entfernt.',
                                                    actionKey: 'delete_library_item',
                                                    onConfirm: () => handleDeleteItem(item.id)
                                                })}
                                            >
                                                Löschen
                                            </button>
                                        </div>
                                        <h4>Analyse-Ergebnis:</h4>
                                        {/* Zeigt die neuen Daten an, wenn sie da sind, sonst den alten Text */}
                                        <pre>{detailViewMode === 'summary' ? generateReadableSummary(item) : item.content}</pre>
                                    </div>
                                )}
                            </div>
                        );
                    })}

                    {libraryTab === 'rhyme_flow' && filteredItems.map(item => {
                        const sourceSong = activeProfile?.library.find(song => song.id === item.sourceLyricId);
                        const displayTitle = sourceSong
                            ? `Reimfluss aus "${sourceSong.title || 'Unbenannter Song'}"`
                            : 'Unbekannter Reimfluss';

                        return (
                            <div key={item.id} className="accordion-item">
                                <div className="accordion-header" onClick={() => setOpenLyricId(openLyricId === item.id ? null : item.id)}>
                                    <span>{displayTitle}</span>
                                    <div className={`accordion-icon ${openLyricId === item.id ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                                </div>
                                {openLyricId === item.id && (
                                    <div className="accordion-content">
                                        <div className="item-actions top">
                                            {/* NEUER BUTTON FÜR OPTIMIERUNG */}
                                            {!item.structuredData && (
                                                 <button 
                                                    className="secondary-action-button" 
                                                    onClick={() => handleOptimizeForDna(item.id)}
                                                    disabled={optimizingItemId === item.id}
                                                >
                                                    {optimizingItemId === item.id ? 'Optimiere...' : 'Für DNA optimieren'}
                                                </button>
                                            )}
                                            <button className="action-button" onClick={() => handleStartEditing(item)}>Bearbeiten</button>
                                            <button 
                                                className="action-button delete" 
                                                onClick={() => requestConfirmation({
                                                    message: 'Reimfluss-Muster wirklich löschen?',
                                                    details: 'Dadurch wird dieses Reimfluss-Muster und alle damit verbundenen Elemente endgültig aus deiner Bibliothek entfernt.',
                                                    actionKey: 'delete_library_item',
                                                    onConfirm: () => handleDeleteItem(item.id)
                                                })}
                                            >
                                                Löschen
                                            </button>
                                        </div>
                                        <h4>Analyse-Ergebnis:</h4>
                                        {/* Zeigt die neuen Daten an, wenn sie da sind, sonst den alten Text */}
                                        <pre>{detailViewMode === 'summary' ? generateReadableSummary(item) : item.content}</pre>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                    </div>
                </div>
            </Panel>
        </>
    );
};

    const renderWriteSongView = () => {
        // Hole die originalen Stil- und Technik-Listen
        const styleOptions = activeProfile?.library.filter(i => i.type === 'style') || [];
        const techniqueOptions = activeProfile?.library.filter(i => i.type === 'technique') || [];

        // NEU: Filtere Duplikate basierend auf dem Inhalt heraus
        const uniqueStyleOptions = styleOptions.filter((option, index, self) =>
            index === self.findIndex((o) => o.content === option.content)
        );

        const uniqueTechniqueOptions = techniqueOptions.filter((option, index, self) =>
            index === self.findIndex((o) => o.content === option.content)
        );

        return (
            <Panel title="Ghostwriter" iconKey="WRITE" description="Wähle die Zutaten für deinen Song: Profil, Beat, Thema und die gewünschten Stil-Elemente.">
                {/* Wende hier die v-stack-Klassen an */}
                <div className="panel-content v-stack v-stack--gap-lg">
                
                        <SectionHeader
                            iconKey="COMPOSITION"
                            title="Song-Komposition"
                            description="Konfiguriere alle Parameter für deinen neuen Song und lass die KI ihn für dich schreiben."
                        />
                    

                    <div className="song-structure">
                    <div className="structure-section">
                        <h5>Musikalische Grundlagen</h5>
                        <div className="form-grid">

                            <div className="form-group">
                                <label>Character / Stil</label>
                                <MultiSelectDropdown
                                    options={uniqueStyleOptions.sort((a, b) => a.content.localeCompare(b.content))}
                                    selectedValues={songStyles}
                                    onToggleOption={handleToggleStyle}
                                    placeholder={`Stile auswählen (${uniqueStyleOptions.length})`}
                                />
                            </div>
                            <div className="form-group">
                                <label>Technik / Skills</label>
                                <MultiSelectDropdown
                                    options={uniqueTechniqueOptions.sort((a, b) => a.content.localeCompare(b.content))}
                                    selectedValues={songTechniques}
                                    onToggleOption={handleToggleTechnique}
                                    placeholder={`Techniken auswählen (${uniqueTechniqueOptions.length})`}
                                />
                            </div>
                        </div>
                    </div>

                    <div className="structure-section">
                        <h5>Beat & Rhythmus</h5>
                        <div className="form-grid">
                            <div className="form-group span-2">
                                <label>Beat-Beschreibung (z.B. "melancholischer Lo-Fi Beat")</label>
                                <textarea
                                    rows={3}
                                    value={songBeatDescription}
                                    onChange={e => setSongBeatDescription(e.target.value)}
                                    className="textarea-input"
                                ></textarea>
                            </div>
                            <div className="form-group">
                                <label>BPM (optional)</label>
                                <input
                                    type="text"
                                    value={songBPM}
                                    onChange={e => setSongBPM(e.target.value)}
                                    className="text-input"
                                />
                            </div>
                            <div className="form-group">
                                <label>Tonart (optional)</label>
                                <input
                                    type="text"
                                    value={songKey}
                                    onChange={e => setSongKey(e.target.value)}
                                    className="text-input"
                                />
                            </div>
                            <div className="button-container">
                                <button 
                                    className="secondary-action-button" 
                                    onClick={() => beatFileInputRef.current?.click()}
                                    disabled={isAnalyzing} // Deaktivieren während der Analyse
                                >
                                    {isAnalyzing ? 'Analysiere Beat...' : 'Beat-Datei analysieren'}
                                </button>
                                <input
                                    type="file"
                                    ref={beatFileInputRef}
                                    onChange={handleBeatFileAnalysis} // Diese Funktion haben wir gerade erstellt
                                    style={{ display: 'none' }}
                                    accept="audio/*"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="structure-section">
                        <h5>Struktur des Textes</h5>
                        <div className="form-grid">
                            <div className="form-group">
                                <label>Songteil</label>
                                <select 
                                    value={songPartType} 
                                    onChange={e => setSongPartType(e.target.value)}
                                    className="form-input-field"
                                >
                                    <option value="full_song">Ganzer Song (2 Parts & Hook)</option>
                                    <option value="part_16">Part (16 Zeilen)</option>
                                    <option value="part_12">Part (12 Zeilen)</option>
                                    <option value="part_8">Part (8 Zeilen)</option>
                                    <option value="hook_8">Hook (8 Zeilen)</option>
                                    <option value="bridge_8">Bridge (8 Zeilen)</option>
                                    <option value="bridge_4">Bridge (4 Zeilen)</option>
                                    <option value="custom">Spezialfall (Zeilenanzahl manuell)</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label>Anzahl der Zeilen (nur für Spezialfall)</label>
                                <input
                                    type="number"
                                    value={numLines}
                                    onChange={e => setNumLines(parseInt(e.target.value, 10))}
                                    className="form-input-field number-input-small" // <-- NEUE KLASSE
                                    disabled={songPartType !== 'custom'} // Deaktiviert, wenn nicht "Spezialfall"
                                    min="1"
                                    max="50"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="structure-section">
                        <h5>Performance & Thema</h5>
                        <div className="form-grid">
                            <div className="form-group">
                                <label>Performance-Stil</label>
                                <select 
                                    value={songPerformanceStyle} 
                                    onChange={e => setSongPerformanceStyle(e.target.value)}
                                    className="form-input-field"
                                >
                                    <option value="Gerappt">Gerappt</option>
                                    <option value="Gesungen">Gesungen</option>
                                    {/* Neue Option, die automatisch bei "Hook" ausgewählt werden kann */}
                                    <option value="Hook (Gesungen)">Hook (Gesungen)</option> 
                                </select>
                            </div>
                            <div className="form-group">
                                <label>Thema des Songs</label>
                                <input
                                    type="text"
                                    value={songTopic}
                                    onChange={e => setSongTopic(e.target.value)}
                                    className="text-input"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="button-container">
                        <button className="main-action-button" onClick={handleGenerateSongText}>
                            Song generieren
                        </button>
                    </div>

                    {/* Anzeige des generierten Songtextes */}
                    {generatedSongText && (
                        <div className="structure-section">
                            <h5>Generierter Songtext</h5>
                            <div className="generated-song-output" onMouseUp={handleTextSelection}>
                                {generatedSongText.split('\n').map((line, index) => (
                                    <p key={index}>{line || '\u00A0'}</p> // Zeilenumbrüche beibehalten
                                ))}
                            </div>
                        </div>
                    )}
                    
                    {generatedSongText && (
                        <div className="button-container generated-song-actions">
                            <button className="secondary-action-button" onClick={handleSaveGeneratedSong}>
                                In Bibliothek speichern
                            </button>
                            <button className="main-action-button" onClick={handleGenerateSongText}>
                                Neu generieren
                            </button>
                        </div>
                    )}
                </div>
                </div>
            </Panel>
        );
    };





const renderRhymeMachineView = () => (
    <Panel title="Rhyme Machine" iconKey="RHYME" description="Die KI findet Reime basierend auf deinen gelernten Regeln.">
        <div className="v-stack v-stack--gap-md">

            {/* NEUE POSITION: Dieses Feature steht jetzt ganz oben */}
            <div className="sub-panel">
                <SectionHeader
                    iconKey="RHYME_TRAINER"
                    title="Bringe der KI deine Reime bei"
                    description="Je mehr Reime du hier eingibst, desto besser versteht die KI deine persönliche Reimlogik. Diese Lektionen fließen in die gesamte Künstler-DNA ein."
                />
                <div className="rhyme-lesson-input">
                    <input
                        type="text"
                        placeholder="Wort / Phrase"
                        value={lessonWord}
                        onChange={e => setLessonWord(e.target.value)}
                    />
                    <span>reimt sich auf</span>
                    <input
                        type="text"
                        placeholder="Dein Reim"
                        value={lessonRhyme}
                        onChange={e => setLessonRhyme(e.target.value)}
                    />
                </div>
                <button className="secondary-action-button align-start" onClick={handleSaveRhymeLesson}>Lektion zur DNA hinzufügen</button>
            </div>

            {/* Das erste Feature von vorher, jetzt an zweiter Stelle */}
            <div className="sub-panel">
                <SectionHeader
                    iconKey="RHYME_SEARCH"
                    title="1. Wort eingeben & Reime finden"
                    description="Gib ein Wort ein und lass die KI passende Reime finden, die deinen persönlichen Regeln entsprechen."
                />
                <div className="rhyme-step-container">
                    <input
                        type="text"
                        placeholder="z.B. Geschichte"
                        value={rhymeInput}
                        onChange={e => {
                            setRhymeInput(e.target.value);
                            setRhymeResults([]);
                            setSearchPerformed(false);
                        }}
                    />
                    {isFindingRhymes ? (
                        <>
                            <div className="loading-bar">
                                <div className="loading-progress"></div>
                            </div>
                            <button className="danger-action-button" onClick={handleAbortAnalysis}>
                                Suche abbrechen
                            </button>
                        </>
                    ) : (
                        <button
                            className="main-action-button"
                            onClick={handleFindRhymesWithAnalysis}
                        >
                            Reime finden
                        </button>
                    )}
                </div>
                <div className="output-container rhyme-results">
                    {isFindingRhymes ? (
                        <div className="typing-indicator" style={{ marginTop: '20px' }}>
                            <div className="typing-dot"></div>
                            <div className="typing-dot"></div>
                            <div className="typing-dot"></div>
                        </div>
                    ) : rhymeResults.length > 0 ? (
                        rhymeResults.map((result, index) => {
                            // Hilfsfunktion, um zu prüfen, ob der Reim schon gespeichert ist
                            const isAlreadySaved = activeProfile?.library
                                .filter(item => item.type === 'rhyme_lesson_group' && (item as RhymeLessonGroup).targetWord === rhymeInput)
                                .some(group => (group as RhymeLessonGroup).rhymes.some(r => r.rhymingWord === result.rhyme));

                            return (
                                <div key={index} className="rhyme-result-item">
                                    <div className="rhyme-content">
                                        <h4>{result.rhyme}</h4>
                                    </div>
                                    <div className="rhyme-actions">
                                        {isAlreadySaved ? (
                                            <span className="save-status-chip">✓ Gespeichert</span>
                                        ) : (
                                            <button 
                                                className="action-button add-rhyme-btn" 
                                                onClick={() => handleSaveSingleRhyme(rhymeInput, result.rhyme)}
                                            >
                                                + Hinzufügen
                                            </button>
                                        )}
                                    </div>
                                </div>
                            );
                        })
                    ) : searchPerformed ? (
                        <p className="placeholder">Keine passenden Reime gefunden, die den Regeln entsprechen.</p>
                    ) : (
                        <p className="placeholder">Deine Reim-Ergebnisse erscheinen hier...</p>
                    )}
                </div>
            </div>

            {/* Das zweite Feature von vorher, jetzt an dritter Stelle */}
            <div className="sub-panel">
                <SectionHeader
                    iconKey="RHYME_GENERATOR"
                    title="Reimzeilen-Generator"
                    description="Gib eine ganze Zeile ein, um qualitativ hochwertige Reimzeilen zu generieren, die alle unsere bewährten Regeln für Sinnhaftigkeit, Kreativität und phonetische Resonanz befolgen."
                />
                <div className="rhyme-line-input">
                    <div className="input-group">
                        <input
                            type="text"
                            className="main-input"
                            placeholder="z.B. Gegen den Wind und jede Regel der Kunst"
                            value={multiRhymeInput}
                            onChange={e => setMultiRhymeInput(e.target.value)}
                        />
                    </div>
                    <div className="form-group">
                        <label>Anzahl der Vorschläge:</label>
                        <input
                            type="number"
                            className="form-input-field number-input-small"
                            value={numLinesToGenerate}
                            onChange={e => setNumLinesToGenerate(parseInt(e.target.value, 10))}
                            min="1"
                            max="10"
                        />
                    </div>
                    <div className="button-container">
                        <button 
                            className="main-action-button" 
                            onClick={handleGenerateMultiRhymes}
                            disabled={!multiRhymeInput.trim() || isGeneratingMultiRhymes}
                        >
                            {isGeneratingMultiRhymes ? 'Generiere...' : 'Reimzeilen generieren'}
                        </button>
                    </div>
                </div>
                
                {/* Ergebnisse der Reimzeilen-Generierung */}
                <div className="rhyme-line-results">
                    {isGeneratingMultiRhymes ? (
                        <div className="loading-container">
                            <div className="typing-indicator">
                                <div className="typing-dot"></div>
                                <div className="typing-dot"></div>
                                <div className="typing-dot"></div>
                            </div>
                            <p className="loading-text">Generiere Reimzeilen...</p>
                        </div>
                    ) : multiRhymeResults.length > 0 ? (
                        <div className="rhyme-line-output">
                            <div className="results-header">
                                <h4>Generierte Reimzeilen</h4>
                                <span className="results-count">{multiRhymeResults.length} Zeilen</span>
                            </div>
                            <div className="results-grid">
                                {multiRhymeResults.map((result, index) => (
                                    <div key={index} className="rhyme-line-item">
                                        <div className="rhyme-line-header">
                                            <span className="line-number">Zeile {index + 1}</span>
                                            <div className="line-actions">
                                                <button 
                                                    className="action-btn copy-btn" 
                                                    title="Kopieren"
                                                    onClick={() => {
                                                        navigator.clipboard.writeText(result.line);
                                                        showStatus('Zeile in die Zwischenablage kopiert!', false);
                                                    }}
                                                >
                                                    📋
                                                </button>
                                            </div>
                                        </div>
                                        <div className="rhyme-line-content">
                                            {result.line}
                                        </div>
                                        {result.explanation && (
                                            <div className="rhyme-line-explanation">
                                                <span className="explanation-icon">📊</span>
                                                {result.explanation}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="empty-state">
                            <div className="empty-icon">🎵</div>
                            <p className="empty-text">Deine generierten Reimzeilen erscheinen hier...</p>
                            <p className="empty-hint">Gib eine Zeile ein und klicke auf "Reimzeilen generieren"</p>
                        </div>
                    )}
                </div>
            </div>

        </div>
    </Panel>
);

const renderKuenstlerDnaView = () => {
    // Hilfsfunktion zum Komprimieren langer Texte
    const compressLongText = (text: string, maxLength: number = 150): string => {
        if (text.length <= maxLength) return text;
        
        // Versuche, den Text an Satzenden zu kürzen
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        let compressed = '';
        
        for (const sentence of sentences) {
            const trimmedSentence = sentence.trim();
            if ((compressed + trimmedSentence).length <= maxLength) {
                compressed += (compressed ? '. ' : '') + trimmedSentence;
            } else {
                break;
            }
        }
        
        // Falls immer noch zu lang, kürze auf Wortbasis
        if (compressed.length === 0 || compressed.length > maxLength) {
            const words = text.split(' ');
            compressed = words.slice(0, Math.floor(maxLength / 8)).join(' '); // Durchschnittliche Wortlänge ~8 Zeichen
        }
        
        return compressed + (compressed.length < text.length ? '...' : '');
    };

    // Kategorien für die DNA-Auswahl
    const categories = [
        { key: 'lyric', label: 'Gelernte Lyrics', items: activeProfile?.library.filter(item => item.type === 'lyric') || [] },
        { key: 'style', label: 'Stil', items: activeProfile?.library.filter(item => item.type === 'style') || [] },
        { key: 'technique', label: 'Technik', items: activeProfile?.library.filter(item => item.type === 'technique') || [] },
        { key: 'emphasis', label: 'Betonung', items: activeProfile?.library.filter(item => item.type === 'emphasis') || [] },
        { key: 'rhyme_flow', label: 'Reimfluss', items: activeProfile?.library.filter(item => item.type === 'rhyme_flow') || [] },
        { key: 'rule_category', label: 'Gelernte Regeln', items: activeProfile?.library.filter(item => item.type === 'rule_category') || [] }
    ];

    const handleAddToDna = () => {
        const selectedIds = Object.keys(pendingDnaSelection).filter(id => pendingDnaSelection[id]);
        if (selectedIds.length > 0) {
            handleUpdateActiveDna(selectedIds);
            setPendingDnaSelection({});
            showStatus(`${selectedIds.length} Elemente zur Künstler-DNA hinzugefügt!`, false);
        } else {
            showStatus('Bitte wähle mindestens ein Element aus.', true);
        }
    };

    const handleRemoveFromDna = (itemId: string) => {
        handleRemoveFromActiveDna(itemId);
        showStatus('Element aus der Künstler-DNA entfernt.', false);
    };

    const handleToggleSelection = (itemId: string) => {
        setPendingDnaSelection(prev => ({
            ...prev,
            [itemId]: !prev[itemId]
        }));
    };

    return (
        <Panel title="Künstler DNA" iconKey="DNA" description="Stelle deine persönliche Künstler-DNA manuell zusammen, indem du die wichtigsten Elemente aus deiner Bibliothek auswählst.">
            <div className="v-stack v-stack--gap-lg">

                {/* NEUE POSITION: Dieses Feature steht jetzt ganz oben */}
                <div className="content-block">
                    <div className="active-dna-section v-stack v-stack--gap-md">
                        <SectionHeader
                            iconKey="DEFAULT"
                            title={`Aktive Künstler-DNA (${activeDnaItemIds.length} Elemente)`}
                            description="Dies sind die aktuell ausgewählten Elemente, die bei der nächsten Generierung als Regelwerk dienen."
                        />
                        <div className="v-stack v-stack--gap-lg">
                            <div className="content-block">
                                <div className="tag-container">
                                    {activeDnaItemIds.length === 0 ? (
                                        <p style={{ color: 'var(--subtle-text-color)', fontSize: '0.9rem' }}>
                                            Deine Künstler-DNA ist noch leer. Wähle oben Elemente aus, um sie hinzuzufügen.
                                        </p>
                                    ) : (
                                        activeDnaItemIds.map(itemId => {
                                            const item = activeProfile?.library.find(libItem => libItem.id === itemId);
                                            if (!item) return null;
                                            
                                            const content = item.type === 'lyric' ? item.title || 'Unbenannter Song' : 
                                                          item.type === 'rule_category' ? item.categoryTitle :
                                                          item.content;

                                            return (
                                                <div key={itemId} className="tag-item">
                                                    <span className="tag-content">{content}</span>
                                                    <button 
                                                        className="tag-remove-btn" 
                                                        title="Entfernen"
                                                        onClick={() => handleRemoveFromActiveDna(itemId)}
                                                    >
                                                        {UI_ICONS.CLOSE}
                                                    </button>
                                                </div>
                                            );
                                        })
                                    )}
                                </div>
                            </div>

                            {activeDnaItemIds.length > 0 && (
                                <div className="content-block">
                                    <div className="dna-clear-actions">
                                        <button 
                                            className="danger-action-button" 
                                            onClick={() => requestConfirmation({
                                                message: 'Wirklich alle DNA-Elemente löschen?',
                                                details: `Diese Aktion entfernt ${activeDnaItemIds.length} Element(e) aus deiner aktiven DNA und kann nicht rückgängig gemacht werden.`,
                                                actionKey: 'clear_all_dna',
                                                onConfirm: handleClearActiveDna // Die eigentliche Löschfunktion wird hier übergeben
                                            })}
                                        >
                                            Alle DNA-Elemente löschen ({activeDnaItemIds.length})
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Dieses Feature ist jetzt an zweiter Stelle */}
                <div className="content-block v-stack v-stack--gap-md">
                    <SectionHeader
                        iconKey="DNA_ADD"
                        title="Elemente zur DNA hinzufügen"
                        description="Wähle aus deiner Bibliothek Elemente aus, um sie zu deiner Künstler-DNA hinzuzufügen."
                    />
                    <div className="dna-selection-container">
                        {categories.map(category => (
                            <div key={category.key} className="accordion-item">
                                <div className="accordion-header" onClick={() => setOpenDnaCategoryKey(openDnaCategoryKey === category.key ? null : category.key)}>
                                    <span>{category.label} ({category.items.length})</span>
                                    <div className={`accordion-icon ${openDnaCategoryKey === category.key ? 'open' : ''}`}>{UI_ICONS.CHEVRON_DOWN}</div>
                                </div>
                                {openDnaCategoryKey === category.key && (
                                    <div className="accordion-content open">
                                        <div className="item-list-container v-stack v-stack--gap-xs"> 
                                            {category.items.length === 0 ? (
                                                <p className="empty-category">Keine Elemente in dieser Kategorie vorhanden.</p>
                                            ) : (
                                                category.items.map(item => (
                                                    <div key={item.id} className="multi-select-item"> 
                                                        <label className="checkbox-label">
                                                            <input
                                                                type="checkbox"
                                                                checked={pendingDnaSelection[item.id] || false}
                                                                onChange={() => handleToggleSelection(item.id)}
                                                            />
                                                            <span className="dna-item-content">
                                                                {(() => {
                                                                    // Diese Funktion entscheidet, welcher Text angezeigt wird
                                                                    if (item.type === 'lyric') return (item as LibraryItem).title;
                                                                    if (item.type === 'rule_category') return (item as RuleCategory).categoryTitle;
                                                                    
                                                                    // NEU: Für Betonung & Reimfluss die Zusammenfassung anzeigen
                                                                    if (item.type === 'emphasis' || item.type === 'rhyme_flow') {
                                                                        return generateReadableSummary(item as LibraryItem);
                                                                    }
                                                                    
                                                                    // Fallback für alle anderen Typen
                                                                    return (item as LibraryItem).content;
                                                                })()}
                                                            </span>
                                                        </label>
                                                    </div>
                                                ))
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                        <div className="dna-actions">
                            <button 
                                className="main-action-button" 
                                onClick={handleAddToDna}
                                disabled={Object.keys(pendingDnaSelection).filter(id => pendingDnaSelection[id]).length === 0}
                            >
                                Auswahl zur DNA hinzufügen
                            </button>
                        </div>
                    </div>
                </div>

                {/* Dieses Feature ist jetzt an dritter Stelle */}
                <div className="content-block">
                    <div className="active-dna-section v-stack v-stack--gap-md">
                        <SectionHeader
                            iconKey="DNA_RHYMES"
                            title="Gelernte Reime zur DNA hinzufügen"
                            description="Füge ein Paket von gelernten Reimen hinzu, um die Reimqualität der KI zu verbessern."
                        />
                        
                        <div className="dna-actions">
                            <button className="main-action-button" onClick={handleAddRecentRhymesToDna}>
                                Letzte 20 Reime hinzufügen
                            </button>
                            <button 
                                className="danger-action-button" 
                                onClick={() => requestConfirmation({
                                    message: 'Alle DNA-Reime wirklich löschen?',
                                    details: `Diese Aktion entfernt ${activeRhymeDnaIds.length} Reim(e) aus deiner aktiven DNA und kann nicht rückgängig gemacht werden.`,
                                    actionKey: 'clear_dna_rhymes',
                                    onConfirm: handleClearRhymesFromDna
                                })}
                                disabled={activeRhymeDnaIds.length === 0}
                            >
                                DNA-Reime löschen ({activeRhymeDnaIds.length})
                            </button>
                        </div>

                        <div className="dna-filter-section">
                            <input 
                                type="text" 
                                className="form-input-field"
                                placeholder="Filter nach Oberbegriff (z.B. 'Liebe')"
                                value={rhymeFilterKeyword}
                                onChange={e => setRhymeFilterKeyword(e.target.value)}
                            />
                            <button className="secondary-action-button" onClick={handleAddFilteredRhymesToDna} disabled={!rhymeFilterKeyword.trim()}>
                                Gefilterte Reime hinzufügen
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </Panel>
    );
};

    // Wir erstellen jetzt für jede Ansicht eine "memoized" Version.
    // Das bedeutet, sie wird nur neu gerendert, wenn sich ihre EIGENEN Eingabefelder ändern.

const renderViewContent = () => {
    switch(currentView) {
        case 'analyze': return renderAnalyzeView();
        case 'trainer': return renderTrainerView();
        case 'rhyme_machine': return renderRhymeMachineView();
        case 'write_song': return renderWriteSongView();
        case 'manage_library': return renderLibraryView();
        case 'kuenstler_dna': return renderKuenstlerDnaView();
        default: return renderAnalyzeView();
    }
};

    // HAUPT-ROUTER der App
    if (currentView === 'start') return renderStartScreen();
    if (currentView === 'transition') return renderTransitionScreen();

    return (
        <div className="app-layout">
            {/* LoadingOverlay hat höchste Priorität und überlagert alles */}
            {isAnalyzing && <LoadingOverlay message={loadingMessage || "Analysiere..."} />}
            
            {renderModals()}
            {statusMessage.text && <div className={`status-message ${statusMessage.isError ? 'error' : 'success'}`}>{statusMessage.text}</div>}

            {renderSideMenu()}
            <main className="content-wrapper">
                {renderAppHeader()}
                <div className="view-content">
                    {renderViewContent()}
                </div>
            </main>
        </div>
    );
};



    // Wir erstellen jetzt für jede Ansicht eine "memoized" Version.
    // Das bedeutet, sie wird nur neu gerendert, wenn sich ihre EIGENEN Eingabefelder ändern.

// === ANWENDUNG STARTEN ===
const container = document.getElementById('root');
if (container) {
    const root = createRoot(container);
    root.render(<React.StrictMode><App /></React.StrictMode>);
}
