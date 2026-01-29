import { useState, useCallback } from "react";
import { Upload, FileSpreadsheet, AlertCircle, Database, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { SAMPLE_DATASETS, getSampleDataset } from "@/data/sampleDatasets";
import type { DatasetState } from "@/pages/DataAgent";
import * as XLSX from "xlsx";

interface DataUploadProps {
  onDataLoaded: (data: DatasetState) => void;
}

const DataUpload = ({ onDataLoaded }: DataUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const { user } = useAuth();

  const parseCSV = (text: string): Record<string, unknown>[] => {
    const lines = text.trim().split("\n");
    if (lines.length < 2) return [];
    
    const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ''));
    const data: Record<string, unknown>[] = [];
    
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(",").map(v => v.trim().replace(/^"|"$/g, ''));
      const row: Record<string, unknown> = {};
      headers.forEach((header, index) => {
        const value = values[index] || '';
        // Try to parse numbers
        const num = parseFloat(value);
        row[header] = !isNaN(num) && value !== '' ? num : value;
      });
      data.push(row);
    }
    
    return data;
  };

  const parseJSON = (text: string): Record<string, unknown>[] => {
    const parsed = JSON.parse(text);
    return Array.isArray(parsed) ? parsed : [parsed];
  };

  const parseExcel = async (file: File): Promise<Record<string, unknown>[]> => {
    const buffer = await file.arrayBuffer();
    const workbook = XLSX.read(buffer, { type: 'array' });
    const firstSheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[firstSheetName];
    const jsonData = XLSX.utils.sheet_to_json(worksheet);
    return jsonData as Record<string, unknown>[];
  };

  const processFile = async (file: File) => {
    if (!user) {
      toast.error("Please sign in to upload data");
      return;
    }

    setIsLoading(true);
    try {
      const text = await file.text();
      let data: Record<string, unknown>[];
      
      const fileName = file.name.toLowerCase();
      
      if (fileName.endsWith('.csv')) {
        data = parseCSV(text);
      } else if (fileName.endsWith('.json')) {
        data = parseJSON(text);
      } else if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) {
        data = await parseExcel(file);
      } else {
        throw new Error("Unsupported file format. Please upload CSV, Excel, or JSON files.");
      }

      if (data.length === 0) {
        throw new Error("No data found in file");
      }

      const columns = Object.keys(data[0]);
      const datasetName = file.name.replace(/\.(csv|json)$/i, '');

      // Save to database
      const { data: savedDataset, error } = await supabase
        .from('datasets')
        .insert([{
          name: datasetName,
          original_filename: file.name,
          raw_data: JSON.parse(JSON.stringify(data)),
          columns: JSON.parse(JSON.stringify(columns)),
          row_count: data.length,
          column_count: columns.length,
          file_size: file.size,
          status: 'uploaded',
          user_id: user.id
        }])
        .select()
        .single();

      if (error) throw error;

      onDataLoaded({
        id: savedDataset.id,
        name: datasetName,
        rawData: data,
        columns,
        status: "uploaded"
      });

      toast.success(`Loaded ${data.length} rows from ${file.name}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to parse file");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  }, [user]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const handleLoadSampleData = async (sampleId: string) => {
    // Check if we're in demo mode
    const isDemoMode = new URLSearchParams(window.location.search).get("demo") === "true";
    
    if (!user && !isDemoMode) {
      toast.error("Please sign in to load sample data");
      return;
    }

    setLoadingSample(sampleId);
    try {
      const sample = getSampleDataset(sampleId);
      if (!sample) {
        throw new Error("Sample dataset not found");
      }

      // In demo mode, skip database save and load directly
      if (isDemoMode) {
        onDataLoaded({
          id: `demo-${sampleId}`,
          name: sample.name,
          rawData: sample.data,
          columns: sample.columns,
          status: "uploaded"
        });
        toast.success(`Loaded ${sample.data.length.toLocaleString()} rows from ${sample.name} (Demo Mode)`);
        return;
      }

      // Save to database (authenticated mode)
      const { data: savedDataset, error } = await supabase
        .from('datasets')
        .insert([{
          name: sample.name,
          original_filename: `${sample.id}-sample.json`,
          raw_data: JSON.parse(JSON.stringify(sample.data)),
          columns: JSON.parse(JSON.stringify(sample.columns)),
          row_count: sample.data.length,
          column_count: sample.columns.length,
          file_size: JSON.stringify(sample.data).length,
          status: 'uploaded',
          user_id: user.id
        }])
        .select()
        .single();

      if (error) throw error;

      onDataLoaded({
        id: savedDataset.id,
        name: sample.name,
        rawData: sample.data,
        columns: sample.columns,
        status: "uploaded"
      });

      toast.success(`Loaded ${sample.data.length.toLocaleString()} rows from ${sample.name}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to load sample data");
    } finally {
      setLoadingSample(null);
    }
  };

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Drop Zone */}
      <div
        data-onboarding="upload-zone"
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-xl p-6 sm:p-12
          transition-all duration-300 cursor-pointer
          ${isDragging 
            ? 'border-primary bg-primary/10 scale-[1.02]' 
            : 'border-border/50 bg-card/30 hover:border-primary/50 hover:bg-card/50'
          }
        `}
      >
        <input
          type="file"
          accept=".csv,.json,.xlsx,.xls"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isLoading}
        />
        
        <div className="flex flex-col items-center text-center space-y-3 sm:space-y-4">
          <div className={`
            w-12 h-12 sm:w-16 sm:h-16 rounded-xl sm:rounded-2xl flex items-center justify-center
            ${isDragging ? 'bg-primary text-primary-foreground' : 'bg-muted'}
            transition-colors duration-300
          `}>
            {isLoading ? (
              <div className="w-6 h-6 sm:w-8 sm:h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            ) : (
              <Upload className="w-6 h-6 sm:w-8 sm:h-8" />
            )}
          </div>
          
          <div>
            <h3 className="text-lg sm:text-xl font-semibold mb-1 sm:mb-2">
              {isDragging ? "Drop your file here" : "Upload your data"}
            </h3>
            <p className="text-sm text-muted-foreground">
              Drag & drop your file, or tap to browse
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs sm:text-sm text-muted-foreground">
            <FileSpreadsheet className="w-4 h-4" />
            <span>Supports CSV, Excel, and JSON</span>
          </div>
        </div>
      </div>

      {/* Sample Datasets */}
      <div data-onboarding="sample-button" className="bg-card/50 border border-border/50 rounded-xl p-4 sm:p-6">
        <div className="flex items-start gap-3 mb-4">
          <Database className="w-5 h-5 text-primary mt-0.5 shrink-0" />
          <div>
            <h4 className="font-medium mb-1 text-sm sm:text-base">Try Sample Datasets</h4>
            <p className="text-xs sm:text-sm text-muted-foreground">
              No data ready? Load a sample dataset to explore features.
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3">
          {SAMPLE_DATASETS.map((sample) => (
            <Button
              key={sample.id}
              variant="outline"
              className="justify-start h-auto py-2.5 sm:py-3 px-3 sm:px-4"
              onClick={() => handleLoadSampleData(sample.id)}
              disabled={loadingSample !== null}
            >
              <div className="flex items-center gap-2 sm:gap-3 w-full">
                {loadingSample === sample.id ? (
                  <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4 text-primary shrink-0" />
                )}
                <div className="text-left flex-1 min-w-0">
                  <p className="font-medium text-xs sm:text-sm truncate">{sample.name}</p>
                  <p className="text-[10px] sm:text-xs text-muted-foreground">{sample.rowCount.toLocaleString()} rows</p>
                </div>
              </div>
            </Button>
          ))}
        </div>
      </div>

      {/* Tips */}
      <div className="bg-card/50 border border-border/50 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-primary mt-0.5" />
          <div>
            <h4 className="font-medium mb-1">Tips for best results</h4>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• Ensure your CSV has headers in the first row</li>
              <li>• Excel files: First sheet will be imported</li>
              <li>• JSON should be an array of objects with consistent keys</li>
              <li>• Virtual scrolling enabled for datasets with 100K+ rows</li>
              <li>• Your data is saved securely and linked to your account</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataUpload;
